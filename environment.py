from utils import Utils
from controller import Controller
from nonlinear_model import NonlinearModel

u = Utils()
controller = Controller()


class Env:
    def __init__(self):
        self.info = ""
        self.dtau = 0.01
        self.dt = 0.001
        self.Nsolver = int(self.dtau / self.dt)

        self.obs_space = 3
        self.action_space = 3
        self.action_space_max = 30
        self.action_space_min = -30
        self.states = self.__initial_states()
        self.obs_states = self.__obs_calc()

        self.action_to_controller = [0, 0, 0, u.feetTometer(176)]
        self.index = 0
        # self.errsum=[0,0,0,0]
        # self.lasterr=[0,0,0,0]

    def reset(self):
        self.states = self.__initial_states()
        self.obs_states = self.__obs_calc()
        self.action_to_controller = [0, 0, 0, u.feetTometer(176)]
        self.index = 0
        # self.errsum=[0,0,0,0]
        # self.lasterr=[0,0,0,0]
        return self.obs_states

    def step(self, action):
        converted_action = self.__action_converter(action)
        next_states = self.__make_action(converted_action)

        self.states = next_states
        self.obs_states = self.__obs_calc()
        done = self.__done_calc()
        reward = self.__reward_calc()
        self.index+=1

        return self.obs_states, reward, done,{}

    def __initial_states(self):
        return [u.feetTometer(176), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4500.0, 0.0, 0.0, 0.0]

    def __obs_calc(self):

        phi = u.radTodeg(self.states[3])
        theta = u.radTodeg(self.states[4])
        psi = u.radTodeg(self.states[5])

        obs = [phi, theta, psi]

        return obs

    def __action_converter(self, action):
        #action = action[0]
        for i in range(3):
            self.action_to_controller[i] = u.degTorad(action[i])
        return self.action_to_controller

    def __make_action(self, actions):
        new_states = self.states
        for i in range(10):
            U = controller.BacksteppingController(new_states, actions)  # self.errsum,self.lasterr,self.dtau
            # U=data[0]
            # self.errsum=data[1]
            # self.lasterr=data[2]
            new_states = NonlinearModel(new_states, U, self.Nsolver, self.dt)
        return new_states

    def __done_calc(self):
        donef = False

        phi = u.radTodeg(self.states[3])
        theta = u.radTodeg(self.states[4])
        psi = u.radTodeg(self.states[5])
        
        if abs(phi)>60 or abs(theta)>60 or abs(psi)>60:
            donef=True


        if self.index == 50:
            donef = True
        return donef

    def __reward_calc(self):
        phi = u.radTodeg(self.states[3])
        theta = u.radTodeg(self.states[4])
        psi = u.radTodeg(self.states[5])
        reward=-1*(abs(phi)+abs(theta)+abs(psi))
        return reward

