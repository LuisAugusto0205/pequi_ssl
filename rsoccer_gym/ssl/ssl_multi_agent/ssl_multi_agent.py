import numpy as np
from gymnasium.spaces import Box, Dict
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
import random
from collections import namedtuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import OrderedDict
class SSLMultiAgentEnv(SSLBaseEnv, MultiAgentEnv):
    default_players = 3
    def __init__(self, n_robots_yellow=0, n_robots_blue=1, field_type=2, 
        init_pos = {'blue': [
            [np.random.uniform(-3, 3), np.random.uniform(-2, 2)],
            [-2, 1],
            [-2, -1],
        ],
        'yellow': [
            [1.5, 0],
            [2, 1],
            [2, -1],
        ]
        },
        ball = [0, 0], max_ep_length=300, options=None, idx=None, r=0.3):
        field = 0 # SSL Division A Field
        agent_ids_blue = [f'blue_{i}'for i in range(n_robots_blue)]
        agent_ids_yellow = [f'yellow_{i}'for i in range(n_robots_yellow)]
        self._agent_ids = [*agent_ids_blue, *agent_ids_yellow]
        self.max_ep_length = max_ep_length
        super().__init__(field_type=field_type, n_robots_blue=n_robots_blue,
                         n_robots_yellow=n_robots_yellow, time_step=0.025, max_ep_length=max_ep_length)
        self.n_robots_yellow = n_robots_yellow
        self.n_robots_blue = n_robots_blue
        self.options=options
        self.idx=idx
        self.r=0.3

        self.ball_dist_scale = np.linalg.norm([self.field.width, self.field.length/2])

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.kick_speed_x = 5.0

        self.init_pos = init_pos
        self.ball = ball

        obs, _ = self.reset()
        if self.n_robots_blue > 0:
            self.obs_size = obs[f'blue_0'].shape[0] # Ball x,y and Robot x, y
        else:
            self.obs_size = obs[f'yellow_0'].shape[0]
        self.act_size = 4

        blue = {f'blue_{i}': Box(low=-self.field.length/2, high=self.field.length/2, shape=(self.act_size, ), dtype=np.float64) for i in range(n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=-self.field.length/2, high=self.field.length/2, shape=(self.act_size, ), dtype=np.float64) for i in range(n_robots_yellow)}
        self.action_space =  Dict(**blue, **yellow)

        blue = {f'blue_{i}': Box(low=-self.field.length/2, high=self.field.length/2, shape=(self.obs_size, ), dtype=np.float64) for i in range(n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=-self.field.length/2, high=self.field.length/2, shape=(self.obs_size, ), dtype=np.float64) for i in range(n_robots_yellow)}
        self.observation_space = Dict(**blue, **yellow)

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

    def _calculate_reward_and_done(self):

        done = {f'blue_{i}':False for i in range(self.n_robots_blue)}
        done.update({f'yellow_{i}':False for i in range(self.n_robots_yellow)})
        done.update({'__all__': False})
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
                r_off = (self._get_3dots_angle_between(blue_robot, ball, goal_adv)[2]/np.pi) - 1.0
                r_def = (self._get_3dots_angle_between(goal_ally, blue_robot, ball)[2]/np.pi) - 1.0

                blue_rw[idx] = 0*r_speed + 0*r_off + 0*r_def

            blue_rw += 1*r_dist*np.ones(self.n_robots_blue)
            blue_rw_dict = {f'blue_{id}':rw for id, rw in enumerate(blue_rw)}
        #print(f'\rr_speed: {r_speed:.2f}\tr_dist: {r_dist:.2f}\tr_off: {r_off:.2f}\tr_def: {r_def:.2f}\ttotal: {0.7*r_speed + 0.1*r_off + 0.1*r_def + 0.1*r_dist:.2f}\t')

        if self.n_robots_yellow > 0:
            yellow_rw = np.zeros(max(self.n_robots_yellow, 1))
            r_dist = -2.5
            for idx in range(self.n_robots_yellow):
                yellow_robot = self.frame.robots_yellow[idx]
                goal_adv = Goal(x=-0.2-self.field.length/2, y=0)
                goal_ally = Goal(x=0.2+self.field.length/2, y=0)

                r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
                r_dist = max(self.__ball_dist_robot_rw(ball, yellow_robot), r_dist)
                r_off = (self._get_3dots_angle_between(yellow_robot, ball, goal_adv)[2]/np.pi)- 1.0
                r_def = (self._get_3dots_angle_between(goal_ally, yellow_robot, ball)[2]/np.pi) - 1.0
                yellow_rw[idx] = 0*r_speed + 0*r_off + 0*r_def

            yellow_rw += 1*r_dist*self.n_robots_yellow
            yellow_rw_dict = {f'yellow_{id}':rw for id, rw in enumerate(yellow_rw)}

        half_len = self.field.length/2 
        #half_wid = self.field.width/2
        half_goal_wid = self.field.goal_width / 2
        if ball.x >= half_len and abs(ball.y) < half_goal_wid:
            done = {f'blue_{i}':True for i in range(self.n_robots_blue)}
            done.update({f'yellow_{i}':True for i in range(self.n_robots_yellow)})
            done.update({'__all__': True})

            blue_rw_dict = {f'blue_{id}': 1 for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{id}': -1 for i in range(self.n_robots_yellow)}
        
        elif ball.x <= -half_len and abs(ball.y) < half_goal_wid:
            done = {f'blue_{i}':True for i in range(self.n_robots_blue)}
            done.update({f'yellow_{i}':True for i in range(self.n_robots_yellow)})
            done.update({'__all__': True})

            blue_rw_dict = {f'blue_{id}': -1 for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{id}': 1 for i in range(self.n_robots_yellow)}

        reward = {**blue_rw_dict, **yellow_rw_dict}
        
        return reward, done
    
        
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
        
        return -ball_dist
        #return -np.clip(ball_dist, 0, 1)
    
    def __ball_grad_rw(self, ball, last_ball, goal):
        assert(self.last_frame is not None)
        
        goal = np.array([goal.x, goal.y])
        # Calculate previous ball dist
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(goal - last_ball_pos)
        
        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal - ball_pos)
        
        ball_dist_rw = last_ball_dist - ball_dist - 0.05
        
        if ball_dist_rw > 1:
            print("ball_dist -> ", ball_dist_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")
        
        return np.clip(ball_dist_rw, -1, 1)

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

        return self._frame_to_observations(True), {**blue, **yellow}
  
    def _get_initial_positions_frame(self, seed):
        '''Returns the position of each robot and ball for the initial frame'''
        np.random.seed(seed)

        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        pos_frame: Frame = Frame()

        if self.idx is None:
            pos_frame.ball = Ball(x=self.ball[0], y=self.ball[1]) #Ball(x=x(), y=y())
        else:
            t = np.linspace(0, 2, 360)
            rx, ry = self.init_pos['blue'][self.idx] if self.idx < self.n_robots_blue else self.init_pos['yellow'][self.idx]
            c_balls = np.vstack([rx + self.r*np.cos(t*np.pi), ry + self.r*np.sin(t*np.pi)]).T
            mask =(abs(c_balls[:, 0]) < field_half_length) & (abs(c_balls[:, 1]) < field_half_width)
            pos_c_balls = c_balls[mask]
            bidx = np.random.randint(pos_c_balls.shape[0])

            pos_frame.ball = Ball(x=pos_c_balls[bidx, 0], y=pos_c_balls[bidx, 1])

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y()) #self.init_pos['blue'][i] 
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=-30)#theta())
            

        for i in range(self.n_robots_yellow):
            pos = self.init_pos['yellow'][i] #(x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=180)#theta())

        return pos_frame

    def _get_pos(self, obj):
        x = self.norm_pos(obj.x)
        y = self.norm_pos(obj.y)
        v_x = self.norm_v(self.frame.ball.v_x)
        v_y = self.norm_v(self.frame.ball.v_y)

        theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
        sin = np.sin(theta) if theta else None
        cos = np.cos(theta) if theta else None
        theta = (theta/np.pi) - 1 if theta else None
        #tan = np.tan(theta) if theta else None

        return x, y, v_x, v_y, sin, cos, theta

    def _get_3dots_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])
        p3 = np.array([obj3.x, obj3.y])

        vec1 = p1 - p2
        vec2 = p3 - p2

        theta = np.arccos(np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        return np.sin(theta), np.cos(theta), theta

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.field.length, self.field.width])
        dist = np.linalg.norm(diff_vec)

        return dist / max_dist

    def _frame_to_observations(self, reset=False):

        observation = {}

        for i in range(self.n_robots_blue):


            robot = self.frame.robots_blue[i] 
            allys = [self.frame.robots_blue[j] for j in range(self.n_robots_blue) if j != i]
            advs = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow)]

            robot_obs = self.robot_observation(robot, allys, advs)
            observation[f'blue_{i}'] = robot_obs

        for i in range(self.n_robots_yellow):
            robot = self.frame.robots_yellow[i] 
            allys = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow) if j != i]
            advs = [self.frame.robots_blue[j] for j in range(self.n_robots_blue)]
            
            robot_obs = self.robot_observation(robot, allys, advs)
            observation[f'yellow_{i}'] = robot_obs

        return observation
    
    def robot_observation(self, robot, allys, adversaries):

        positions = []
        orientations = []
        dists = []
        angles = []

        ball = self.frame.ball
        goal = namedtuple('goal', ['x', 'y'])
        goal_adv = goal(x=0.2 + self.field.length/2, y=0)
        goal_ally = goal(x=-0.2 - self.field.length/2, y=0)

        x_b, y_b, *_ = self._get_pos(ball)
        sin_BG_al, cos_BG_al, theta_BG_al = self._get_2dots_angle_between(goal_ally, ball)
        sin_BG_ad, cos_BG_ad, theta_BG_ad = self._get_2dots_angle_between(goal_adv, ball)
        dist_BG_al = self._get_dist_between(ball, goal_ally)
        dist_BG_ad = self._get_dist_between(ball, goal_adv)

        x_r, y_r, *_, sin_r, cos_r, theta_r  = self._get_pos(robot)
        sin_BR, cos_BR, theta_BR = self._get_2dots_angle_between(ball, robot)
        dist_BR = self._get_dist_between(ball, robot)

        positions.append([x_r, y_r])
        orientations.append([sin_r, cos_r, theta_r])
        dists.append([dist_BR, dist_BG_al, dist_BG_ad])
        angles.append([
            sin_BR, cos_BR, theta_BR/np.pi, 
            sin_BG_al, cos_BG_al, theta_BG_al/np.pi, 
            sin_BG_ad, cos_BG_ad, theta_BG_ad/np.pi
        ])

        for ally in allys:
            x_al, y_al, *_, sin_al, cos_al, theta_al = self._get_pos(ally)
            sin_AlR, cos_AlR, theta_AlR = self._get_2dots_angle_between(ally, robot)
            ally_dist = self._get_dist_between(ally, robot)
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR/np.pi])
        
        for i in range(self.default_players - len(allys)):
            x_al, y_al, sin_al, cos_al, theta_al = 0, 0, 0, 0, 0
            sin_AlR, cos_AlR, theta_AlR = 0, 0, 0
            ally_dist = 0
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al/np.pi])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR/np.pi])

        
        for adv in adversaries:
            x_adv, y_adv, *_,  sin_adv, cos_adv, theta_adv = self._get_pos(adv)
            sin_AdR, cos_AdR, theta_AdR = self._get_2dots_angle_between(adv, robot)
            adv_dist = self._get_dist_between(adv, robot)
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv/np.pi])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR/np.pi])

        for i in range(self.default_players - len(adversaries)):
            x_adv, y_adv, sin_adv, cos_adv, theta_adv = 0, 0, 0, 0, 0
            sin_AdR, cos_AdR, theta_AdR = 0, 0, 0
            adv_dist = 0
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv/np.pi])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR/np.pi])

        positions.append([x_b, y_b])

        positions = np.concatenate(positions)
        orientations = np.concatenate(orientations)
        dists = np.concatenate(dists)
        angles = np.concatenate(angles)
        time_left = [(self.max_ep_length - self.steps)/self.max_ep_length]

        robot_obs = np.concatenate([positions, orientations, dists, angles, time_left], dtype=np.float64)
        return robot_obs
    
    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        if self.steps >= self.max_ep_length:
            done = {f'blue_{i}':True for i in range(self.n_robots_blue)}
            done.update({f'yellow_{i}':True for i in range(self.n_robots_yellow)})
            done.update({'__all__': True})
        
        blue = {f'blue_{i}': {} for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}':{} for i in range(self.n_robots_yellow)}

        return observation, reward, done, done, {**blue, **yellow}
        
if __name__ == "__main__":
    print(SSLMultiAgentEnv.mro())
