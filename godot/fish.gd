extends Node3D

# Reference to the PathFollow3D node
var path_follow: PathFollow3D

# Speed parameters
@export var target_speed: float = 15.0  # Target movement speed
@export var inertia: float = 0.95      # Higher = more inertia (0.0-1.0)

# Current speed (affected by inertia)
var current_speed: float = 0.0

func _ready():
	# Get reference to the PathFollow3D node
	path_follow = $Path3D2/PathFollow3D
	
	# Optional: Make sure the path follow is at the start
	path_follow.progress = 0.0
	
	# Ensure loop is enabled for continuous movement
	path_follow.loop = true

func _process(delta):
	current_speed = current_speed + (target_speed - current_speed) * (1.0 - inertia)
	
	path_follow.progress += delta * current_speed
