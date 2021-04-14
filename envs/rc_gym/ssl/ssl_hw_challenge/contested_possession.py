import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot, Ball
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLContestedPossessionEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal with contested possession


        Description:
            The episode starts with the enemy robot with ball possession 
            blocking the goal, the controlled robot needs to score without
            breaking division B rules
        Observation:
            Type: Box(4 + 7*n_robots_blue + 2*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->10    id 0 Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            +2*i     id i Yellow Robot [X, Y]
        Actions:
            Type: Box(5, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
            3       id 0 Blue Kick x Speed  (%)
            4       id 0 Blue Dribbler  (%) (true if % is positive)
            
        Reward:
            1 if goal
        Starting State:
            Enemy robot with ball possession facing away from goal
        Episode Termination:
            Goal, 30 seconds (1200 steps), or rule infraction
    """
    def __init__(self, random_init=False):
        super().__init__(field_type=2, n_robots_blue=1, 
                         n_robots_yellow=1, time_step=0.025)
        self.random_init = random_init
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(5, ), dtype=np.float32)
        
        n_obs = 4 + 8*self.n_robots_blue + 2*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        # scale max dist rw to 1 Considering that max possible move rw if ball and robot are in opposite corners of field
        self.ball_dist_scale = np.linalg.norm([self.field.width, self.field.length/2])
        self.ball_grad_scale = np.linalg.norm([self.field.width/2, self.field.length/2])/4
        
        # scale max energy rw to 1 Considering that max possible energy if max robot wheel speed sent every step
        wheel_max_rad_s = 160
        max_steps = 1200
        self.energy_scale = ((wheel_max_rad_s * 4) * max_steps)

        print('Environment initialized')

    def reset(self):
        self.reward_shaping_total = None
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))
            observation.append(1 if self.frame.robots_blue[i].infrared else 0)

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        cmd = Robot(yellow=False, id=0, v_x=actions[0],
                              v_y=actions[1], v_theta=actions[2],
                              kick_v_x=1. if actions[3] > 0 else 0., 
                              dribbler=True if actions[4] > 0 else False)
        cmd.to_local(self.frame.robots_blue[0].theta)
        commands.append(cmd)
        
        return commands

    def _calculate_reward_and_done(self):
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                'goal': 0,
                'rbt_in_gk_area': 0,
                'done_ball_out': 0,
                'done_ball_out_right': 0,
                'done_rbt_out': 0,
                'ball_dist': 0,
                'ball_grad': 0,
                'energy': 0,
                'collision': 0
            }
        reward = 0
        done = False
        
        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2
        
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid
        
        # End episode in case of collision
        for rbt in self.frame.robots_yellow.values():
            if abs(rbt.v_x) > 0.1 or abs(rbt.v_y) > 0.1:
                self.reward_shaping_total['collision'] += 1
                done = True
        
        # Check if robot exited field right side limits
        if robot.x < -0.2 or abs(robot.y) > half_wid:
            done = True
            self.reward_shaping_total['done_rbt_out'] += 1
        # If flag is set, end episode if robot enter gk area
        elif robot_in_gk_area(robot):
            done = True
            self.reward_shaping_total['rbt_in_gk_area'] += 1
        # Check ball for ending conditions
        elif ball.x < 0 or abs(ball.y) > half_wid:
            done = True
            self.reward_shaping_total['done_ball_out'] += 1
        elif ball.x > half_len:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = 5 
                self.reward_shaping_total['goal'] += 1
            else:
                reward = 0
                self.reward_shaping_total['done_ball_out_right'] += 1
        elif self.last_frame is not None:
            ball_dist_rw = self.__ball_dist_rw() / self.ball_dist_scale
            self.reward_shaping_total['ball_dist'] += ball_dist_rw
            
            ball_grad_rw = self.__ball_grad_rw() / self.ball_grad_scale
            self.reward_shaping_total['ball_grad'] += ball_grad_rw
            
            energy_rw = -self.__energy_pen() / self.energy_scale
            self.reward_shaping_total['energy'] += energy_rw
            
            reward = reward\
                    + ball_dist_rw\
                    + ball_grad_rw\
                    + energy_rw

        done = done

        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        pos_frame: Frame = Frame()
        def x(): return random.uniform(pen_len, half_len - pen_len)
        def y(): return random.uniform(-half_pen_wid, half_pen_wid)
        
        if self.random_init:
            pos_frame.robots_blue[0] = Robot(x=x()-pen_len, y=y(), theta=0.)
            enemy_x = x()
            enemy_y = y()
        else:
            pos_frame.robots_blue[0] = Robot(x=0, y=0, theta=0.)
            enemy_x = x()
            enemy_y = y()


        pos_frame.ball = Ball(x=enemy_x-0.1, y=enemy_y)
        pos_frame.robots_yellow[0] = Robot(x=enemy_x, y=enemy_y, theta=180.)

        return pos_frame

    def __ball_dist_rw(self):
        assert(self.last_frame is not None)
        
        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        last_robot = self.last_frame.robots_blue[0]
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_ball_dist = np.linalg.norm(last_robot_pos - last_ball_pos)
        
        # Calculate new ball dist
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
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

    def __ball_grad_rw(self):
        assert(self.last_frame is not None)
        
        # Goal pos
        goal = np.array([self.field.length/2, 0.])
        
        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        ball = self.frame.ball
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(goal - last_ball_pos)
        
        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal - ball_pos)
        
        ball_dist_rw = last_ball_dist - ball_dist
        
        if ball_dist_rw > 1:
            print("ball_dist -> ", ball_dist_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")
        
        return np.clip(ball_dist_rw, -1, 1)

    def __energy_pen(self):
        robot = self.frame.robots_blue[0]
        
        # Sum of abs each wheel speed sent
        energy = abs(robot.v_wheel0)\
            + abs(robot.v_wheel1)\
            + abs(robot.v_wheel2)\
            + abs(robot.v_wheel3)
            
        return energy
