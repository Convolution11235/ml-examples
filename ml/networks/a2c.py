import torch
import torch.nn as nn
import gaime.networks.features as ft
import functools as ftools
import torch.nn.functional as fn
import math

from torch.distributions import Categorical
from dataclasses import dataclass

@dataclass
class Params:
	features: ft.CVParams = ft.CVParams()
	# Can be various values from ft. e.g. ft.LSTMParams
	features_type: object = None
	layers: int = 1
	act: object = lambda i: torch.nn.ReLU()




class A2C_Vis(nn.Module):
	"""
	Sample A2C (Advantage Actor Critic) network that takes images for its input
	and generates output.
	"""

	def __init__(self, in_size, out_size, params=Params()):
		super().__init__()
		self.__params = params
		self.__in_size = in_size
		self.__out_size = out_size

		
		self.__features = ft.Image2d(
			in_size, 
			params=params.features,
			features=params.features_type)

		# Create a linearly decreasing size of linear layers.
		self.__actor = nn.Sequential()
		s = ftools.reduce(lambda x, a: x*a, self.__features.out_size, 1)
		sl = (out_size - s)/params.layers
		v = lambda x: math.ceil(sl * x + s)
		for i in range(params.layers):
			self.__actor.add_module(f"Actor-Linear {i}", nn.Linear(v(i), v(i+1)))
			self.__actor.add_module(f"Actor-Activation {i}", params.act(i))


		self.__critic = nn.Sequential()
		sl = (1 - s)/params.layers
		v = lambda x: math.ceil(sl * x + s)
		for i in range(params.layers):
			self.__critic.add_module(f"Critic-Linear {i}", nn.Linear(v(i), v(i+1)))
			self.__critic.add_module(f"Critic-Activation {i}", params.act(i))




	def forward(self, x):
		x = self.__features(x).flatten(start_dim=1)
		return self.__actor(x), self.__critic(x)

	def action(self, x):
		state = self(x)[0]

		distr = fn.softmax(state, dim=1) 
		choice = Categorical(distr).sample()
		return (choice, distr)
		


