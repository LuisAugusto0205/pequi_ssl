import numpy as np
from gymnasium.spaces import Box, Dict
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
import random
from collections import namedtuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import OrderedDict

GOAL_REWARD = 10

class SSLMultiAgentEnv(SSLBaseEnv, MultiAgentEnv):
    default_players = 3
    def __init__(self,
        init_pos,
        field_type=2, 
        fps=40,
        match_time=40,
        stack_observation=8

    ):

        self.n_robots_yellow = 3
        self.n_robots_blue = 3
        self.score = {'blue': 0, 'yellow': 0}
        super().__init__(
            field_type=field_type, 
            n_robots_blue=self.n_robots_blue,
            n_robots_yellow=self.n_robots_yellow, 
            time_step=1/fps, 
            max_ep_length=int(match_time*fps),
        )

        agent_ids_blue = [f'blue_{i}'for i in range(self.n_robots_blue)]
        agent_ids_yellow = [f'yellow_{i}'for i in range(self.n_robots_yellow)]
        self._agent_ids = [*agent_ids_blue, *agent_ids_yellow]
        self.max_ep_length = int(match_time*fps)
        self.fps = fps
        self.last_actions = {
            **{f'blue_{i}': np.zeros(4) for i in range(self.n_robots_blue)}, 
            **{f'yellow_{i}': np.zeros(4) for i in range(self.n_robots_yellow)}
        }

        self.stack_observation = stack_observation
        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.kick_speed_x = 5.0

        self.init_pos = init_pos

        self.obs_size = 77 #obs[f'blue_0'].shape[0]
        self.act_size = 4

        self.actions_bound = {"low": -1, "high": 1}

        blue = {f'blue_{i}': Box(low=self.actions_bound["low"], high=self.actions_bound["high"], shape=(self.act_size, ), dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=self.actions_bound["low"], high=self.actions_bound["high"], shape=(self.act_size, ), dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.action_space =  Dict(**blue, **yellow)

        blue = {f'blue_{i}': Box(low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(self.obs_size, ), dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(self.obs_size, ), dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.observation_space = Dict(**blue, **yellow)

        self.observations = {
            **{f'blue_{i}': np.zeros(self.stack_observation * self.obs_size) for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(self.stack_observation * self.obs_size) for i in range(self.n_robots_yellow)}
        }

    def _get_commands(self, actions):
        commands = []
        for i in range(self.n_robots_blue):
            robot_actions = actions[f'blue_{i}'].copy()
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, angle)
            cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)
        
        for i in range(self.n_robots_yellow):
            robot_actions = actions[f'yellow_{i}'].copy()
            angle = self.frame.robots_yellow[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, angle)

            cmd = Robot(yellow=True, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)

        return commands
    
    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""

        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _calculate_reward_done_info(self):

        # done = {f'blue_{i}':False for i in range(self.n_robots_blue)}
        # done.update({f'yellow_{i}':False for i in range(self.n_robots_yellow)})
        done = {'__all__': False}
        ball = self.frame.ball
        last_ball = self.last_frame.ball

        Goal = namedtuple('goal', ['x', 'y'])
        blue_rw_dict = {}
        yellow_rw_dict = {}

        if self.n_robots_blue > 0:
            blue_rw = np.zeros(self.n_robots_blue)
            r_dist = -2.5
            for idx in range(self.n_robots_blue):
                blue_robot = self.frame.robots_blue[idx]
                goal_adv = Goal(x=0.2+self.field.length/2, y=0)
                goal_ally = Goal(x=-0.2-self.field.length/2, y=0)

                r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
                r_dist = max(self.__ball_dist_robot_rw(ball, blue_robot), r_dist)
                r_off = self._get_3dots_angle_between(blue_robot, ball, goal_adv)[2] - 1
                r_def = self._get_3dots_angle_between(goal_ally, blue_robot, ball)[2] - 1

                blue_rw[idx] = 0.7*r_speed + 0.1*r_off + 0.1*r_def

            blue_rw += 0.1*r_dist*np.ones(self.n_robots_blue)
            blue_rw_dict = {f'blue_{id}':rw for id, rw in enumerate(blue_rw)}
        print(f'\rr_speed: {r_speed:.2f}\tr_dist: {r_dist:.2f}\tr_off: {r_off:.2f}\tr_def: {r_def:.2f}\ttotal: {0.7*r_speed + 0.1*r_off + 0.1*r_def + 0.1*r_dist:.2f}\t')

        if self.n_robots_yellow > 0:
            yellow_rw = np.zeros(max(self.n_robots_yellow, 1))
            r_dist = -2.5
            for idx in range(self.n_robots_yellow):
                yellow_robot = self.frame.robots_yellow[idx]
                goal_adv = Goal(x=-0.2-self.field.length/2, y=0)
                goal_ally = Goal(x=0.2+self.field.length/2, y=0)

                r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
                r_dist = max(self.__ball_dist_robot_rw(ball, yellow_robot), r_dist)
                r_off = self._get_3dots_angle_between(yellow_robot, ball, goal_adv)[2] - 1
                r_def = self._get_3dots_angle_between(goal_ally, yellow_robot, ball)[2] - 1
                yellow_rw[idx] = 0.7*r_speed + 0.1*r_off + 0.1*r_def

            yellow_rw += 0.1*r_dist*self.n_robots_yellow
            yellow_rw_dict = {f'yellow_{id}':rw for id, rw in enumerate(yellow_rw)}

        half_len = self.field.length/2 
        #half_wid = self.field.width/2
        half_goal_wid = self.field.goal_width / 2

        if ball.x >= half_len and abs(ball.y) < half_goal_wid:
            # done = {f'blue_{i}':True for i in range(self.n_robots_blue)}
            # done.update({f'yellow_{i}':True for i in range(self.n_robots_yellow)})
            done = {'__all__': True}
            self.score['blue'] += 1

            blue_rw_dict = {f'blue_{i}': GOAL_REWARD for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{i}': -GOAL_REWARD for i in range(self.n_robots_yellow)}
        
        elif ball.x <= -half_len and abs(ball.y) < half_goal_wid:
            # done = {f'blue_{i}':True for i in range(self.n_robots_blue)}
            # done.update({f'yellow_{i}':True for i in range(self.n_robots_yellow)})
            done = {'__all__': True}
            self.score['yellow'] += 1

            blue_rw_dict = {f'blue_{i}': -GOAL_REWARD for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{i}': GOAL_REWARD for i in range(self.n_robots_yellow)}
        
        infos = {
            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
        }
        if done.get("__all__", False):
            for i in range(self.n_robots_blue):
                infos[f'blue_{i}']["score"] = self.score.copy()

            for i in range(self.n_robots_yellow):
                infos[f'yellow_{i}']["score"] = self.score.copy()

        reward = {**blue_rw_dict, **yellow_rw_dict}
        
        return reward, done, infos
    
        
    def __ball_dist_rw(self, ball, last_ball, robot, last_robot):
        assert(self.last_frame is not None)
        
        # Calculate previous ball dist
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_ball_dist = np.linalg.norm(last_robot_pos - last_ball_pos)
        
        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])
        ball_dist = np.linalg.norm(robot_pos - ball_pos)
        
        ball_dist_rw = last_ball_dist - ball_dist
        
        if ball_dist_rw > 1:
            print("ball_dist -> ", ball_dist_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")
        
        return np.clip(ball_dist_rw, -1, 1)

    def __ball_dist_robot_rw(self, ball, robot):
        assert(self.last_frame is not None)
        
        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])
        ball_dist = np.linalg.norm(robot_pos - ball_pos)
        
        #return -ball_dist
        return -np.clip(ball_dist, 0, 1)
    
    def __ball_grad_rw(self, ball, last_ball, goal):
        assert(self.last_frame is not None)
        
        goal = np.array([goal.x, goal.y])
        # Calculate previous ball dist
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(goal - last_ball_pos)
        
        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal - ball_pos)
        diff = last_ball_dist - ball_dist
        ball_dist = min(diff - 0.01, 2*self.max_v*(1/self.fps))
        ball_speed_rw = ball_dist/(1/self.fps)
        
        if ball_speed_rw > 1:
            print("ball_dist -> ", ball_speed_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")
        
        return np.clip(ball_speed_rw, -1, 1)

    def reset(self, seed=42, options={}):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        # Close render window
        del(self.view)
        self.view = None

        initial_pos_frame: Frame = self._get_initial_positions_frame(seed)
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        blue = {f'blue_{i}': {} for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}':{} for i in range(self.n_robots_yellow)}

        self._frame_to_observations()

        return self.observations.copy(), {**blue, **yellow}
  
    def _get_initial_positions_frame(self, seed):
        '''Returns the position of each robot and ball for the initial frame'''
        #np.random.seed(seed)

        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)
        
        def theta(): return random.uniform(0, 360)

        places = KDTree()

        pos_frame: Frame = Frame()
        pos_frame.ball = Ball(x=0, y=0)
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        min_dist = 0.2
        for i in range(self.n_robots_blue):
            pos = self.init_pos['blue'][i+1] 
            while places.get_nearest(pos[:2])[1] < min_dist:
                pos = (x(), y(), theta()) 
            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])
            

        for i in range(self.n_robots_yellow):
            pos = self.init_pos['yellow'][i+1] 
            while places.get_nearest(pos[:2])[1] < min_dist:
                pos = (x(), y(), theta()) 

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])

        return pos_frame

    def _get_pos(self, obj):

        x = self.norm_pos(obj.x)
        y = self.norm_pos(obj.y)
        v_x = self.norm_v(obj.v_x)
        v_y = self.norm_v(obj.v_y)

        theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
        sin = np.sin(theta) if theta else None
        cos = np.cos(theta) if theta else None
        theta = np.arctan2(sin, cos) if theta else None
        v_theta = self.norm_w(obj.v_theta) if theta else None
        #tan = np.tan(theta) if theta else None

        return x, y, v_x, v_y, sin, cos, theta, v_theta

    def _get_3dots_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])
        p3 = np.array([obj3.x, obj3.y])

        vec1 = p1 - p2
        vec2 = p3 - p2

        theta = np.arccos(np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        return np.sin(theta), np.cos(theta), theta/np.pi

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta/np.pi
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.field.length, self.field.width])
        dist = np.linalg.norm(diff_vec)

        return dist / max_dist

    def _frame_to_observations(self):

        for i in range(self.n_robots_blue):

            robot = self.frame.robots_blue[i] 
            robot_action = self.last_actions[f'blue_{i}']
            allys = [self.frame.robots_blue[j] for j in range(self.n_robots_blue) if j != i]
            allys_actions = [self.last_actions[f'blue_{j}'] for j in range(self.n_robots_blue) if j != i]
            advs = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow)]

            robot_obs = self.robot_observation(robot, allys, advs, robot_action, allys_actions)
            self.observations[f'blue_{i}'] = np.delete(self.observations[f'blue_{i}'], range(len(robot_obs)))
            self.observations[f'blue_{i}'] = np.concatenate([self.observations[f'blue_{i}'], robot_obs], axis=0)

        for i in range(self.n_robots_yellow):
            robot = self.frame.robots_yellow[i]
            robot_action = self.last_actions[f'yellow_{i}']
            allys = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow) if j != i]
            allys_actions = [self.last_actions[f'yellow_{j}'] for j in range(self.n_robots_yellow) if j != i]
            advs = [self.frame.robots_blue[j] for j in range(self.n_robots_blue)]
            
            robot_obs = self.robot_observation(robot, allys, advs, robot_action, allys_actions)
            self.observations[f'yellow_{i}'] = np.delete(self.observations[f'yellow_{i}'], range(len(robot_obs)))
            self.observations[f'yellow_{i}'] = np.concatenate([self.observations[f'yellow_{i}'], robot_obs], axis=0)
    
    def robot_observation(self, robot, allys, adversaries, robot_action, allys_actions):

        positions = []
        orientations = []
        dists = []
        angles = []
        last_actions = np.array([robot_action] + allys_actions).flatten()

        ball = self.frame.ball
        goal = namedtuple('goal', ['x', 'y'])
        goal_adv = goal(x=0.2 + self.field.length/2, y=0)
        goal_ally = goal(x=-0.2 - self.field.length/2, y=0)

        x_b, y_b, *_ = self._get_pos(ball)
        sin_BG_al, cos_BG_al, theta_BG_al = self._get_2dots_angle_between(goal_ally, ball)
        sin_BG_ad, cos_BG_ad, theta_BG_ad = self._get_2dots_angle_between(goal_adv, ball)
        dist_BG_al = self._get_dist_between(ball, goal_ally)
        dist_BG_ad = self._get_dist_between(ball, goal_adv)

        x_r, y_r, *_, sin_r, cos_r, theta_r, _  = self._get_pos(robot)
        sin_BR, cos_BR, theta_BR = self._get_2dots_angle_between(ball, robot)
        dist_BR = self._get_dist_between(ball, robot)

        positions.append([x_r, y_r])
        orientations.append([sin_r, cos_r, theta_r])
        dists.append([dist_BR, dist_BG_al, dist_BG_ad])
        angles.append([
            sin_BR, cos_BR, theta_BR, 
            sin_BG_al, cos_BG_al, theta_BG_al, 
            sin_BG_ad, cos_BG_ad, theta_BG_ad
        ])

        for ally in allys:
            x_al, y_al, *_, sin_al, cos_al, theta_al, _ = self._get_pos(ally)
            sin_AlR, cos_AlR, theta_AlR = self._get_2dots_angle_between(ally, robot)
            ally_dist = self._get_dist_between(ally, robot)
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])
        
        for i in range(self.default_players - len(allys) - 1):
            print("não é pra entrar aqui")
            x_al, y_al, sin_al, cos_al, theta_al = 0, 0, 0, 0, 0
            sin_AlR, cos_AlR, theta_AlR = 0, 0, 0
            ally_dist = 0
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])

        
        for adv in adversaries:
            x_adv, y_adv, *_,  sin_adv, cos_adv, theta_adv, _ = self._get_pos(adv)
            sin_AdR, cos_AdR, theta_AdR = self._get_2dots_angle_between(adv, robot)
            adv_dist = self._get_dist_between(adv, robot)
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        for i in range(self.default_players - len(adversaries)):
            x_adv, y_adv, sin_adv, cos_adv, theta_adv = 0, 0, 0, 0, 0
            sin_AdR, cos_AdR, theta_AdR = 0, 0, 0
            adv_dist = 0
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        positions.append([x_b, y_b])

        positions = np.concatenate(positions)
        orientations = np.concatenate(orientations)
        dists = np.concatenate(dists)
        angles = np.concatenate(angles)
        time_left = [(self.max_ep_length - self.steps)/self.max_ep_length]

        robot_obs = np.concatenate([positions, orientations, dists, angles, last_actions, time_left], dtype=np.float64)
        return robot_obs
    
    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        self.last_actions = action.copy()

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        self._frame_to_observations()
        reward, done, info = self._calculate_reward_done_info()

        if self.steps >= self.max_ep_length:
            # done = {f'blue_{i}':True for i in range(self.n_robots_blue)}
            # done.update({f'yellow_{i}':True for i in range(self.n_robots_yellow)})
            done = {'__all__': True}

        return self.observations.copy(), reward, done, done, info
        
if __name__ == "__main__":
    print(SSLMultiAgentEnv.mro())
