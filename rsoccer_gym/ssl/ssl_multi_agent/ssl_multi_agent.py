import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
import random
from collections import namedtuple



class SSLMultiAgentEnv(SSLBaseEnv):
    def __init__(self, n_robots_yellow=3, n_robots_blue=3, field_type=2, 
        init_pos = {'blue': [
            [-1.5, 0],
            [-2, 1],
            [-2, -1],
        ],
        'yellow': [
            [1.5, 0],
            [2, 1],
            [2, -1],
        ]
        },
        ball = [0, 0], max_ep_length=300):
        field = 0 # SSL Division A Field
        super().__init__(field_type=field_type, n_robots_blue=n_robots_blue,
                         n_robots_yellow=n_robots_yellow, time_step=0.025, max_ep_length=max_ep_length)
        self.n_robots_yellow = n_robots_yellow
        self.n_robots_blue = n_robots_blue

        self.ball_dist_scale = np.linalg.norm([self.field.width, self.field.length/2])

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.kick_speed_x = 5.0

        self.init_pos = init_pos
        self.ball = ball

        obs, _ = self.reset()
        self.obs_size = obs.shape[1] # Ball x,y and Robot x, y
        self.act_size = 4
        self.action_space = Box(low=-1, high=1, shape=(self.act_size*(n_robots_blue + n_robots_yellow), ))
        self.observation_space = Box(low=-self.field.length/2, high=self.field.length/2, shape=(self.obs_size*(n_robots_blue + n_robots_yellow), ))

    def _get_commands(self, actions):
        commands = []
        for i in range(self.n_robots_blue):
            robot_actions = actions[self.act_size*i: self.act_size*(i + 1)]
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, angle)
            cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)
        
        for i in range(self.n_robots_yellow):
            robot_actions = actions[self.act_size*(self.n_robots_blue + i): self.act_size*(self.n_robots_blue + i + 1)]
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

        done=False
        ball = self.frame.ball
        last_ball = self.last_frame.ball

        Goal = namedtuple('goal', ['x', 'y'])

        blue_rw = np.zeros(max(self.n_robots_blue, 1))
        r_dist = -1
        for idx in range(self.n_robots_blue):
            blue_robot = self.frame.robots_blue[idx]
            goal_adv = Goal(x=self.field.length/2, y=0)
            goal_ally = Goal(x=-self.field.length/2, y=0)

            r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
            r_dist = max(self.__ball_dist_robot_rw(ball, blue_robot), r_dist)
            r_off = (np.arccos(self._get_angle_between(blue_robot, ball, goal_adv)[1])/np.pi) - 1.0
            r_def = (np.arccos(self._get_angle_between(goal_ally, blue_robot, ball)[1])/np.pi) - 1.0

            blue_rw[idx] = 0.7*r_speed + 0.1*r_off + 0.1*r_def
        blue_rw += 0.1*r_dist*np.ones(max(self.n_robots_blue, 1))
        #print(f'\rr_speed: {r_speed:.2f}\tr_dist: {r_dist:.2f}\tr_off: {r_off:.2f}\tr_def: {r_def:.2f}\ttotal: {0.7*r_speed + 0.1*r_off + 0.1*r_def + 0.1*r_dist:.2f}\t', end='')

        yellow_rw = np.zeros(max(self.n_robots_yellow, 1))
        r_dist = -1
        for idx in range(self.n_robots_yellow):
            yellow_robot = self.frame.robots_yellow[idx]
            goal_adv = Goal(x=-self.field.length/2, y=0)
            goal_ally = Goal(x=self.field.length/2, y=0)

            r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
            r_dist = max(self.__ball_dist_robot_rw(ball, yellow_robot), r_dist)
            r_off = (np.arccos(self._get_angle_between(yellow_robot, ball, goal_adv)[1])/np.pi)- 1.0
            r_def = (np.arccos(self._get_angle_between(goal_ally, yellow_robot, ball)[1])/np.pi) - 1.0
            yellow_rw[idx] = 0.7*r_speed + 0.1*r_off + 0.1*r_def
        yellow_rw += 0.1*r_dist*np.ones(max(self.n_robots_yellow, 1))

        half_len = self.field.length/2 
        #half_wid = self.field.width/2
        half_goal_wid = self.field.goal_width / 2
        if abs(ball.x) >= half_len and abs(ball.y) > half_goal_wid:
            done = True
        reward = [blue_rw, yellow_rw]
        
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
        
        ball_dist_rw = last_ball_dist - ball_dist - 0.05
        
        if ball_dist_rw > 1:
            print("ball_dist -> ", ball_dist_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")
        
        return np.clip(ball_dist_rw, -1, 1)

    def reset(self, seed=42, options=None, idx=None, r=0.3):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        # Close render window
        del(self.view)
        self.view = None

        initial_pos_frame: Frame = self._get_initial_positions_frame(idx, r, seed)
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        return self._frame_to_observations(), {}
  
    def _get_initial_positions_frame(self, idx=None, r=0.3, seed=None):
        '''Returns the position of each robot and ball for the initial frame'''
        np.random.seed(seed)

        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        pos_frame: Frame = Frame()

        if idx is None:
            pos_frame.ball = Ball(x=self.ball[0], y=self.ball[1]) #Ball(x=x(), y=y())
        else:
            t = np.linspace(0, 2, 360)
            rx, ry = self.init_pos['blue'][idx] if idx < self.n_robots_blue else self.init_pos['yellow'][idx]
            c_balls = np.vstack([rx + r*np.cos(t*np.pi), ry + r*np.sin(t*np.pi)]).T
            mask =(abs(c_balls[:, 0]) < field_half_length) & (abs(c_balls[:, 1]) < field_half_width)
            pos_c_balls = c_balls[mask]
            idx = np.random.randint(pos_c_balls.shape[0])

            pos_frame.ball = Ball(x=pos_c_balls[idx, 0], y=pos_c_balls[idx, 1])

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = self.init_pos['blue'][i] #(x(), y())
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
        y = self.norm_pos(obj.x)
        v_x = self.norm_v(self.frame.ball.v_x)
        v_y = self.norm_v(self.frame.ball.v_y)

        theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
        sin = np.sin(theta) if theta else None
        cos = np.cos(theta) if theta else None
        #tan = np.tan(theta) if theta else None

        return x, y, v_x, v_y, theta, sin, cos

    def _get_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])
        p3 = np.array([obj3.x, obj3.y])

        vec1 = p1 - p2
        vec2 = p3 - p2

        cos = np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        theta = np.arccos(cos)
        sin = np.sin(theta)

        return sin, cos, theta
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2

        return np.linalg.norm(diff_vec)

    def _frame_to_observations(self):

        observation = []

        for i in range(self.n_robots_blue):


            robot = self.frame.robots_blue[i] 
            allys = [self.frame.robots_blue[j] for j in range(self.n_robots_blue) if j != i]
            advs = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow)]

            robot_obs = self.robot_observation(robot, allys, advs)
            observation.append(robot_obs)

        for i in range(self.n_robots_yellow):
            robot = self.frame.robots_yellow[i] 
            allys = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow) if j != i]
            advs = [self.frame.robots_blue[j] for j in range(self.n_robots_blue)]
            
            robot_obs = self.robot_observation(robot, allys, advs)
            observation.append(robot_obs)

        return np.array(observation, dtype=np.float32)

    def robot_observation(self, robot, allys, adversaries):
        robot_obs = []

        ball = self.frame.ball
        goal = namedtuple('goal', ['x', 'y'])
        goal_adv = goal(x=self.field.length/2, y=0)
        goal_ally = goal(x=-self.field.length/2, y=0)

        xb, yb, v_xb, x_yb, *_ = self._get_pos(ball)
        robot_obs.append([xb, yb, v_xb, x_yb]) # 4

        robot_pos = self._get_pos(robot)
        ball_dist = self._get_dist_between(ball, robot)
        robot_obs.append(list(robot_pos)) # 7
        robot_obs.append([ball_dist]) # 1

        angle_ball_goal_ally = self._get_angle_between(goal_ally, robot, ball)
        angle_ball_goal_adv = self._get_angle_between(goal_adv, robot, self.frame.ball)
        robot_obs.append(list(angle_ball_goal_ally)) # 3
        robot_obs.append(list(angle_ball_goal_adv)) # 3

        angle_robot_goal_ally = self._get_angle_between(goal_ally, ball, robot)
        angle_robot_goal_adv = self._get_angle_between(goal_adv, ball, robot)
        robot_obs.append(list(angle_robot_goal_ally)) # 3
        robot_obs.append(list(angle_robot_goal_adv)) # 3

        for ally in allys: 
            ally_pos = self._get_pos(ally)
            ally_dist = self._get_dist_between(ally, robot)
            ally_adv_angles = map(lambda adv: list(self._get_angle_between(ally, robot, adv)), adversaries)
            robot_obs.append(list(ally_pos)) #  7 x 2 = 14
            robot_obs.append([ally_dist]) # 1 x 2 = 2
            for angles in ally_adv_angles:
                robot_obs.append(angles) # 3 x 3 x 2 = 18

        for adv in adversaries:
            adv_pos = self._get_pos(adv)
            adv_dist = self._get_dist_between(adv, robot)
            robot_obs.append(list(adv_pos)) # 7 x 3 = 21
            robot_obs.append([adv_dist]) # 1 x 3 = 3
        
        robot_obs.append([(self.max_ep_length - self.steps)/self.max_ep_length])
        return np.concatenate(robot_obs)
        