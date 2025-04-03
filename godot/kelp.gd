extends Node3D

# Seaweed parameters
@export var sway_speed: float = 1.0
@export var sway_strength: float = 0.1
@export var stiffness: float = 0.8  # How quickly it returns to original position (0-1)
@export var damping: float = 0.9    # How quickly the movement slows down (0-1)
@export var collision_mask: int = 1  # Which layers to check for collisions
@export var segment_length: float = 0.2  # Approximate length between bones

# Internal variables
var bones = []
var velocities = []
var original_transforms = []
var time_offset = 0.0
var skeleton: Skeleton3D
var physics_space_state: PhysicsDirectSpaceState3D

func _ready():
	# Initialize with a random offset for varied movement
	time_offset = randf() * 10.0
	
	# Find the existing skeleton
	skeleton = get_node_or_null("Skeleton3D")
	if skeleton == null:
		skeleton = $"."
		if not skeleton is Skeleton3D:
			push_error("Could not find a Skeleton3D node. Make sure there's a Skeleton3D node as a child or this script is attached to a Skeleton3D.")
			return
	
	# Initialize bones and velocities
	setup_bones()
	
	# Get physics space for collision detection
	physics_space_state = get_world_3d().direct_space_state
	
	# Set up collision detection
	setup_collision_detection()

func setup_bones():
	# Get all the bones from the skeleton
	bones.clear()
	velocities.clear()
	original_transforms.clear()
	
	var bone_count = skeleton.get_bone_count()
	for i in range(bone_count):
		# Only include bones that are part of the seaweed (you might need to adjust this logic)
		var bone_name = skeleton.get_bone_name(i)
		if bone_name.begins_with("Bone") or "seaweed" in bone_name.to_lower():
			bones.append(i)
			velocities.append(Vector3.ZERO)
			original_transforms.append(skeleton.get_bone_rest(i))

func setup_collision_detection():
	# Add collision shapes for physics interaction
	for i in range(1, bones.size()):  # Skip the root bone
		var bone_idx = bones[i]
		var bone_pos = skeleton.get_bone_global_pose(bone_idx).origin
		var parent_idx = skeleton.get_bone_parent(bone_idx)
		var parent_pos = skeleton.get_bone_global_pose(parent_idx).origin
		
		var shape = CollisionShape3D.new()
		var capsule = CapsuleShape3D.new()
		
		# Calculate segment length based on the distance to parent
		var bone_length = bone_pos.distance_to(parent_pos)
		
		capsule.radius = bone_length * 0.15
		capsule.height = bone_length * 0.7
		
		shape.shape = capsule
		
		# Position halfway between this bone and its parent
		var midpoint = (bone_pos + parent_pos) / 2.0
		shape.global_position = midpoint
		
		# Rotate to align with bone direction
		var look_dir = parent_pos - bone_pos
		if look_dir.length() > 0.001:
			shape.look_at(parent_pos, Vector3.UP)
			shape.rotation.x += PI/2  # Adjust to align capsule with bone
		
		var area = Area3D.new()
		area.collision_mask = collision_mask
		area.collision_layer = 0
		area.monitoring = true
		area.monitorable = false
		area.name = "SeaweedSegment_" + str(i)
		
		add_child(area)
		area.add_child(shape)
		
		# Connect the area signals
		area.area_entered.connect(_on_area_entered.bind(i))
		area.body_entered.connect(_on_body_entered.bind(i))

func _physics_process(delta):
	if bones.size() < 2:
		return
		
	update_physics(delta)

func update_physics(delta):
	# The root bone stays fixed
	velocities[0] = Vector3.ZERO
	
	# Natural sway force
	var current_time = Time.get_ticks_msec() / 1000.0 + time_offset
	var sway_force = Vector3(sin(current_time * sway_speed) * sway_strength, 
							 0, 
							 cos(current_time * 0.7 * sway_speed) * sway_strength * 0.7)
	
	# Update each bone (except the root)
	for i in range(1, bones.size()):
		var bone_idx = bones[i]
		var parent_idx = skeleton.get_bone_parent(bone_idx)
		
		# Check if parent is in our processed bones array
		var parent_local_idx = bones.find(parent_idx)
		if parent_local_idx == -1:
			# If parent is not a tracked bone, skip this bone
			continue
		
		# Apply natural sway force (decreases with distance from base)
		var segment_sway = sway_force * (1.0 - float(i) / bones.size() * 0.5)
		velocities[i] += segment_sway * delta
		
		# Apply physics constraints
		var parent_pose = skeleton.get_bone_global_pose(parent_idx)
		var current_pose = skeleton.get_bone_global_pose(bone_idx)
		var parent_pos = parent_pose.origin
		var current_pos = current_pose.origin
		
		# Calculate target position based on rest pose length
		var rest_offset = original_transforms[i].origin - original_transforms[parent_local_idx].origin
		var target_dir = (parent_pos - current_pos).normalized()
		var rest_length = rest_offset.length()
		var target_pos = parent_pos + target_dir * -rest_length
		
		# Apply stiffness (pulls back to ideal position)
		velocities[i] += (target_pos - current_pos) * stiffness * delta * 10.0
		
		# Apply damping (slows movement over time)
		velocities[i] *= damping
		
		# Update position
		var new_pos = current_pos + velocities[i]
		
		# Maintain segment length constraint
		var dir = (new_pos - parent_pos).normalized()
		new_pos = parent_pos + dir * -rest_length
		
		# Update the bone transform
		var new_transform = skeleton.get_bone_global_pose(bone_idx)
		new_transform.origin = new_pos
		
		# Look at parent for better orientation
		var look_at_transform = new_transform.looking_at(parent_pos, Vector3.UP)
		new_transform.basis = look_at_transform.basis
		
		# Set the new bone transform
		skeleton.set_bone_global_pose_override(bone_idx, new_transform, 1.0, true)

func _on_area_entered(area: Area3D, bone_idx: int):
	apply_collision_force(bone_idx, area.global_position, 5.0)

func _on_body_entered(body: Node3D, bone_idx: int):
	apply_collision_force(bone_idx, body.global_position, 10.0)

func apply_collision_force(bone_idx: int, collision_point: Vector3, force: float):
	if bone_idx >= velocities.size():
		return
		
	# Calculate direction away from collision
	var bone_pos = skeleton.get_bone_global_pose(bones[bone_idx]).origin
	var dir = (bone_pos - collision_point).normalized()
	
	# Apply a force to push the bone away
	velocities[bone_idx] += dir * force
	
	# Also affect nearby bones with decreasing intensity
	for i in range(1, 3):
		var nearby_idx = bone_idx + i
		if nearby_idx < velocities.size():
			velocities[nearby_idx] += dir * force * (0.7 / i)
