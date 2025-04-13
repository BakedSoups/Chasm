shader_type spatial;
render_mode cull_disabled, unshaded;

// Wave motion parameters
uniform float wave_speed = 0.2;
uniform float wave_strength = 1.5;
uniform float wave_frequency = 2.0;
uniform float secondary_wave_frequency = 3.0;
uniform float secondary_wave_strength = 0.5;

// How big, in world space, is the noise texture
// current will tile every current_texture_tile_size
uniform float current_texture_tile_size = 20.0;
uniform float vertical_movement = 0.8;
uniform vec2 current_direction = vec2(1.0, 0.5);

// Water depth effect
uniform float depth_influence = 0.3;
uniform float water_density = 1.2;

// Textures
uniform sampler2D color_ramp : source_color;
uniform sampler2D current_noise : hint_default_black;

// Character interaction
uniform vec3 character_position;
uniform float character_radius = 3.0;
uniform sampler2D character_distance_falloff_curve : hint_default_black;
uniform float character_push_strength = 1.0;

// Buoyancy settings
uniform float buoyancy = 0.5;
uniform float bobbing_speed = 0.8;
uniform float bobbing_strength = 0.7;

varying float debug_current;

void vertex() {
    vec3 world_vert = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
    vec2 normalized_current_direction = normalize(current_direction);
    
    // Base current movement
    vec2 primary_uv = world_vert.xz / current_texture_tile_size + normalized_current_direction * TIME * wave_speed;
    float primary_noise = textureLod(current_noise, primary_uv, 0.0).r - 0.5;
    
    // Secondary current for more organic movement (using the same noise texture but different UV)
    vec2 secondary_uv = world_vert.xz / (current_texture_tile_size * 0.5) - normalized_current_direction * TIME * wave_speed * 0.7;
    float secondary_noise = textureLod(current_noise, secondary_uv, 0.0).r - 0.5;
    
    // Combined noise for more natural movement
    float combined_noise = primary_noise * 0.7 + secondary_noise * 0.3;
    
    // Create a main bobbing motion that affects the entire strand
    // This creates the primary left-right swaying
    float main_bob = sin(TIME * bobbing_speed) * bobbing_strength;
    
    // Create a wave that moves up from the base (stem)
    // The sin function creates a wave, and we multiply by UV.y to make it stronger at the top
    float wave_offset = sin(TIME * wave_frequency + world_vert.y * 2.0) * wave_strength;
    float secondary_wave = cos(TIME * secondary_wave_frequency + world_vert.x * 2.0) * secondary_wave_strength;
    
    // The stem movement factor - increases from bottom to top
    // We use a non-linear curve for more natural bending
    float stem_factor = pow(UV.y, 2.0); // Quadratic increase for more natural bending
    
    // Create a line-wide bobbing factor that affects the entire strand
    // This decreases slightly toward the tip to create a natural flow
    float line_bob_factor = 1.0 - (UV.y * 0.3);
    
    // Convert the direction of the current into vertex space from world space
    vec3 vert_space_horizontal_dir = (inverse(MODEL_MATRIX) * vec4(current_direction, 0.0, 0.0)).xyz;
    vert_space_horizontal_dir = normalize(vert_space_horizontal_dir);
    
    // Create the current movement vector
    vec3 current_force = vec3(
        combined_noise * vert_space_horizontal_dir.x + wave_offset + (main_bob * line_bob_factor),
        combined_noise * vertical_movement + secondary_wave, // Add vertical movement for floating effect
        combined_noise * vert_space_horizontal_dir.z + wave_offset + (main_bob * line_bob_factor * 0.5)
    );
    
    // Apply buoyancy - gradual upward force that increases toward the top of the seaweed
    current_force.y += buoyancy * stem_factor;
    
    // Apply depth influence (seaweed moves more at shallow depths)
    float depth_factor = 1.0 - (world_vert.y / (water_density * 10.0));
    depth_factor = clamp(depth_factor, 0.0, 1.0);
    current_force *= mix(1.0, depth_influence, depth_factor);
    
    // Apply the displacement with stem_factor for more natural movement from the base
    VERTEX += current_force * stem_factor;
    
    // Character interaction (similar to the original shader)
    vec3 dir_to_character = character_position - MODEL_MATRIX[3].xyz;
    float distance_to_character = length(dir_to_character);
    float falloff = 1.0 - smoothstep(0.0, 1.0, distance_to_character/character_radius);
    
    // Convert direction to vertex space
    dir_to_character = (inverse(MODEL_MATRIX) * vec4(dir_to_character, 0.0)).xyz;
    dir_to_character = normalize(dir_to_character);
    
    // Apply character interaction force
    float falloff_curve = texture(character_distance_falloff_curve, vec2(falloff)).x;
    VERTEX += normalize(-dir_to_character) * falloff_curve * character_push_strength * stem_factor;
}

void fragment() {
    // Use the original color ramp
    ALBEDO = texture(color_ramp, vec2(1.0 - UV.y, 0)).rgb;
    
    // Optional: Add a slight translucency effect for underwater look
    ALPHA = 0.9; // Slightly translucent
}