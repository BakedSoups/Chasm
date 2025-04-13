extends CharacterBody3D

@export var max_speed = 40.5
@export var acceleration = 12.0
@export var deceleration = 5.0
@export var rotation_speed = 3.0
@export var vertical_rotation_speed = 2.0  # Speed for up/down rotation
@export var max_vertical_rotation = 0.5    # Maximum pitch in radians (about 30 degrees)

@export var dash_speed = 90.0
@export var dash_duration = 0.3
@export var dash_cooldown = 1.0
@export var dash_acceleration = 3.0
@export var dash_deceleration = 1.0
@export var post_dash_inertia = 0.7
@export var camera_sensitivity = 0.003
@export var controller_look_sensitivity = 2.0
@export var spring_length = 5.0
@export var camera_smoothing = 0.1
@export var dash_camera_smoothing = 0.5

@onready var spring_arm = $SpringArm3D
@onready var camera = $SpringArm3D/Camera3D
@onready var mesh = $Skeleton3D
@onready var animation_player = $AnimationPlayer3
@onready var animation_player2 = $AnimationPlayer2
## target search is placed on physics layer 32, and searches for targets on layer 32
@onready var target_search = $TargetSearch

var is_moving = false
var is_dashing = false
var dash_timer = 0.0
var dash_cooldown_timer = 0.0
var observation_mode = false
var last_move_direction = Vector3.ZERO
var dash_direction = Vector3.ZERO
var dash_velocity = Vector3.ZERO
var current_dash_speed = 0.0
var target_dash_speed = 0.0

var target_spring_rotation = Vector3.ZERO
var current_spring_rotation = Vector3.ZERO
var target_mesh_rotation = 0.0
var target_mesh_pitch = 0.0      # Target pitch for up/down rotation
var current_mesh_pitch = 0.0     # Current pitch for up/down rotation

var target_head = null

var current_animation_state = "idle"
var debug_counter = 0.0

func _ready():
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	
	spring_arm.spring_length = spring_length
	spring_arm.top_level = true
	
	current_spring_rotation = spring_arm.rotation
	target_spring_rotation = spring_arm.rotation
	
	_ensure_animations_loop()
	
	_switch_to_idle_animation()

func _input(event):
	if event is InputEventMouseMotion:
		target_spring_rotation.y -= event.relative.x * camera_sensitivity
		
		target_spring_rotation.x -= event.relative.y * camera_sensitivity
		target_spring_rotation.x = clamp(target_spring_rotation.x, -PI/2, PI/2)
	
	if event.is_action_pressed("ui_cancel"):
		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	
	if event.is_action_pressed("target"):
		if target_head:
			target_head = null
		else:
			var closest = closest_entity()
			if closest: target_head = closest.target
		print("Focusing: ", target_head)

func _process(delta):
	var joy_look = Vector2.ZERO
	
	## if targeting something focus on it
	## else, follow the player's input
	if target_head:
		var old_rotation = spring_arm.rotation
		spring_arm.look_at(target_head.global_position)
		var new_rotation = spring_arm.rotation
		
		target_spring_rotation = lerp_rotation(old_rotation, new_rotation, delta * 3)
		target_spring_rotation.x = clamp(target_spring_rotation.x, -PI/2, PI/2)
	else:
		joy_look.x = Input.get_joy_axis(0, JOY_AXIS_RIGHT_X)
		joy_look.y = Input.get_joy_axis(0, JOY_AXIS_RIGHT_Y)
	
		if joy_look.length() < 0.2:
			joy_look = Vector2.ZERO
		
		if joy_look != Vector2.ZERO:
			target_spring_rotation.y -= joy_look.x * controller_look_sensitivity * delta
			target_spring_rotation.x -= joy_look.y * controller_look_sensitivity * delta
			target_spring_rotation.x = clamp(target_spring_rotation.x, -PI/2, PI/2)
	
	_update_animation_state()
	
	if debug_counter != null:
		debug_counter += delta
		if debug_counter > 2.0:
			debug_counter = 0.0
			print("Animation 1 playing: ", animation_player.is_playing(), 
				  " Animation 2 playing: ", animation_player2.is_playing(),
				  " Current state: ", current_animation_state,
				  " Is moving: ", is_moving)

func _physics_process(delta):
	if dash_cooldown_timer > 0:
		dash_cooldown_timer -= delta
	if dash_timer > 0:
		dash_timer -= delta
		if dash_timer <= 0:
			is_dashing = false
	
	_update_camera_position(delta)
	
	var move_direction = Vector3.ZERO
	
	var camera_basis = spring_arm.global_transform.basis
	var forward = -camera_basis.z
	var right = camera_basis.x
	
	forward.y = 0
	forward = forward.normalized()
	right.y = 0
	right = right.normalized()
	
	if Input.is_action_pressed("ui_up"):
		move_direction += forward
	if Input.is_action_pressed("ui_down"):
		move_direction -= forward
	if Input.is_action_pressed("ui_right"):
		move_direction += right
	if Input.is_action_pressed("ui_left"):
		move_direction -= right
	
	var joy_direction = Vector2.ZERO
	joy_direction.x = Input.get_joy_axis(0, JOY_AXIS_LEFT_X)
	joy_direction.y = Input.get_joy_axis(0, JOY_AXIS_LEFT_Y)
	
	if joy_direction.length() < 0.2:
		joy_direction = Vector2.ZERO
	else:
		move_direction += right * joy_direction.x
		move_direction += forward * -joy_direction.y
	
	if Input.is_action_pressed("ui_rise"):
		target_mesh_pitch = -max_vertical_rotation  # Pitch up when rising
	elif Input.is_action_pressed("ui_dive"):
		target_mesh_pitch = max_vertical_rotation   # Pitch down when diving
	else:
		target_mesh_pitch = 0.0  # Return to neutral when not moving vertically
	
	if Input.is_action_just_pressed("ui_dash") and dash_cooldown_timer <= 0 and !is_dashing:
		if last_move_direction.length() > 0.1:
			dash_direction = last_move_direction
		else:
			dash_direction = -camera_basis.z
			dash_direction.y = 0
			dash_direction = dash_direction.normalized()
		
		if dash_direction.length() > 0.1:
			dash_direction.y = 0
			dash_direction = dash_direction.normalized()
			
			is_dashing = true
			dash_timer = dash_duration
			dash_cooldown_timer = dash_cooldown
			
			dash_velocity = velocity
			current_dash_speed = velocity.length()
			target_dash_speed = dash_speed
			
			if velocity.length() < 5.0:
				current_dash_speed = 5.0
	
	if move_direction.length() > 0:
		move_direction = move_direction.normalized()
		last_move_direction = move_direction
		is_moving = true
		
		if observation_mode:
			observation_mode = false
	else:
		is_moving = false
		
		if !is_moving and !observation_mode and !is_dashing:
			observation_mode = true
	
	if is_dashing:
		var dash_progress = 1.0 - (dash_timer / dash_duration)
		
		if dash_progress < 0.5:
			current_dash_speed = lerp(current_dash_speed, target_dash_speed, delta * dash_acceleration)
		else:
			current_dash_speed = lerp(current_dash_speed, max_speed * post_dash_inertia, delta * dash_deceleration)
		
		var dash_move_vector = dash_direction * current_dash_speed
		
		velocity.x = dash_move_vector.x
		velocity.z = dash_move_vector.z
		
		if dash_timer <= 0:
			is_dashing = false
	else:
		var momentum_factor = 1.0
		
		if dash_timer <= 0 and dash_cooldown_timer > dash_cooldown - 0.3:
			momentum_factor = 0.5  # Apply less control right after dash
		
		if is_moving:
			var target_velocity = move_direction * max_speed
			velocity.x = lerp(velocity.x, target_velocity.x, acceleration * delta * momentum_factor)
			velocity.z = lerp(velocity.z, target_velocity.z, acceleration * delta * momentum_factor)
		else:
			velocity.x = lerp(velocity.x, 0.0, deceleration * delta)
			velocity.z = lerp(velocity.z, 0.0, deceleration * delta)
		
		if Input.is_action_pressed("ui_rise"):
			velocity.y = lerp(velocity.y, max_speed, acceleration * delta)
		elif Input.is_action_pressed("ui_dive"):
			velocity.y = lerp(velocity.y, -max_speed, acceleration * delta)
		else:
			velocity.y = lerp(velocity.y, 0.0, deceleration * delta)
			
		# Update last_move_direction with horizontal components only
		var horizontal_velocity = Vector3(velocity.x, 0, velocity.z)
		var last_move_direction = Vector3(velocity.x, 0, velocity.z).normalized()
		if last_move_direction.length() < 0.1:
			last_move_direction = -camera_basis.z
			last_move_direction.y = 0
			last_move_direction = last_move_direction.normalized()
	
	if is_moving or is_dashing:
		var direction_to_face = velocity.normalized() if is_dashing else move_direction
		target_mesh_rotation = atan2(direction_to_face.x, direction_to_face.z)
	elif !observation_mode:
		var camera_direction = -camera_basis.z
		camera_direction.y = 0
		camera_direction = camera_direction.normalized()
		
		if camera_direction.length() > 0.1:
			target_mesh_rotation = atan2(camera_direction.x, camera_direction.z)
	
	_update_mesh_rotation(delta)
	
	move_and_slide()

func _update_camera_position(delta):
	var smoothing_value = dash_camera_smoothing if is_dashing else camera_smoothing
	var smoothing_speed = 1.0 - exp(-smoothing_value / delta)
	
	current_spring_rotation = current_spring_rotation.lerp(target_spring_rotation, smoothing_speed)
	spring_arm.rotation = current_spring_rotation
	
	# Instantly track character position (no lag)
	spring_arm.global_position = global_position
	spring_arm.spring_length = spring_length

func _update_mesh_rotation(delta):
	if observation_mode:
		return
		
	var current_rotation = mesh.rotation.y
	var rotation_diff = fposmod(target_mesh_rotation - current_rotation + PI, TAU) - PI
	var yaw_smoothing_factor = 1.0 - exp(-rotation_speed * delta)
	mesh.rotation.y += rotation_diff * yaw_smoothing_factor
	
	var pitch_smoothing_factor = 1.0 - exp(-vertical_rotation_speed * delta)
	current_mesh_pitch = lerp(current_mesh_pitch, target_mesh_pitch, pitch_smoothing_factor)
	mesh.rotation.x = current_mesh_pitch

func _update_animation_state():
	if is_dashing or is_moving:
		if current_animation_state != "moving":
			current_animation_state = "moving"
			_switch_to_moving_animation()
	else:
		if current_animation_state != "idle":
			current_animation_state = "idle"
			_switch_to_idle_animation()

func _ensure_animations_loop():
	var idle_anim = animation_player2.get_animation("mixamo_com")
	var move_anim = animation_player.get_animation("mixamo_com")
	
	if idle_anim:
		idle_anim.loop_mode = Animation.LOOP_LINEAR
	if move_anim:
		move_anim.loop_mode = Animation.LOOP_LINEAR

func _switch_to_moving_animation():
	animation_player2.stop()
	
	animation_player.stop()
	animation_player.set_speed_scale(1.0) 
	animation_player.play("mixamo_com")
	print("Switched to moving animation - Animation playing: ", animation_player.is_playing(), " Current anim: ", animation_player.current_animation)

func _switch_to_idle_animation():
	animation_player.stop()
	
	animation_player2.stop()
	animation_player2.set_speed_scale(1.0)
	animation_player2.play("mixamo_com")
	print("Switched to idle animation - Animation playing: ", animation_player2.is_playing(), " Current anim: ", animation_player2.current_animation)

func closest_entity():
	var entities = target_search.get_overlapping_bodies()
	var closest = null
	var d = INF
	
	for e in entities:
		var dist = e.global_position.distance_to(self.global_position)
		if dist < d:
			d = dist
			closest = e
	
	return closest

func lerp_rotation(old, new, delta):
	return Vector3(
		lerp_angle(old.x, new.x, delta),
		lerp_angle(old.y, new.y, delta),
		lerp_angle(old.z, new.z, delta),
	)
