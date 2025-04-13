shader_type spatial;
render_mode cull_disabled, blend_mix, depth_prepass_alpha;

uniform sampler2D screen_texture : hint_screen_texture, source_color, repeat_disable;
uniform sampler2D normal_map_texture : hint_normal;
uniform vec3 water_color : source_color = vec3(0.0625, 0.1992, 0.3594);
uniform vec3 underwater_ripple_color : source_color = vec3(0.05, 0.12, 0.25); // Blue color for underwater ripples
uniform float index_of_refraction = 1.333;
uniform float window_scale = 0.5; // Control Snell's window size (smaller value = smaller window)
uniform float transparency = 0.45;
uniform float ripple_strength = 0.02;
uniform float ripple_speed = 0.0005;
uniform float ripple_scale = 4.0; // Higher value = smaller ripples
uniform float underwater_ripple_visibility = .5; // Controls how visible ripples are underwater
uniform float underwater_ripple_tint = 0.2; // Controls blue tint intensity

// Returns 1.0 if we can see above the water, 0.0 otherwise
float snells_window(vec3 normal, vec3 view, float ior) {
    float cos_theta = dot(normal, view);
    // Multiply the IOR by window_scale to make the window smaller
    float scaled_ior = ior * (1.0 / window_scale);
    return step(sqrt(1.0 - cos_theta * cos_theta) * scaled_ior, 1.0);
}

// Computes Fresnel reflection factor for water
float fresnel(float n1, float n2, vec3 normal, vec3 view) {
    float r0 = pow((n1 - n2) / (n1 + n2), 2.0);
    float cosTheta = dot(normal, view);
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}

void fragment() {
    // Material properties
    METALLIC = 0.1;
    ROUGHNESS = 0.05;
    SPECULAR = 0.7;
    
    // Time variables for animation
    float time = TIME * (ripple_speed * 0.3);
    
    // Scale UV coordinates to make ripples smaller
    vec2 scaled_uv = UV * ripple_scale;
    
    // Sample normal map with two different scrolling directions for more complex waves
    // Using scaled UV coordinates for smaller ripples
    vec3 normal_sample_1 = texture(normal_map_texture, scaled_uv + vec2(-1, 1) * time).xyz * 2.0 - 1.0;
    vec3 normal_sample_2 = texture(normal_map_texture, scaled_uv * 1.5 + vec2(1, 0.5) * time).xyz * 2.0 - 1.0;
    
    // Combine normal samples for more complex wave patterns
    vec3 normal_map = vec3(
        normal_sample_1.xy * 0.6 + normal_sample_2.xy * 0.4,
        0.0
    );
    
    // Calculate Z component to ensure the normal is properly normalized
    normal_map.z = sqrt(max(0.0, 1.0 - dot(normal_map.xy, normal_map.xy)));
    
    // Apply normal mapping
    NORMAL_MAP_DEPTH = 1.0;
    NORMAL = normalize(mix(
        NORMAL,
        TANGENT * normal_map.x + BINORMAL * normal_map.y + NORMAL * normal_map.z,
        NORMAL_MAP_DEPTH
    ));
    
    // Adjust distortion based on view distance to prevent extreme values
    float dist_factor = min(1.0, 4.0 / length(VERTEX));
    
    // Screen distortion for refraction effect
    vec2 distorted_uv = SCREEN_UV - normal_map.xy * ripple_strength * dist_factor;
    vec3 refracted_color = texture(screen_texture, distorted_uv).rgb;
    
    // Handle underwater view (back face)
    if (!FRONT_FACING) {
        // Calculate Snell's window mask
        float window_mask = snells_window(NORMAL, VIEW, index_of_refraction);
        
        // Calculate ripple intensity - this will be used to tint the ripples blue
        float ripple_intensity = length(normal_map.xy) * underwater_ripple_visibility;
        
        // Apply ripple distortion to the screen UV
        vec2 underwater_distorted_uv = SCREEN_UV - normal_map.xy * ripple_strength * dist_factor * underwater_ripple_visibility;
        vec3 underwater_refracted = texture(screen_texture, underwater_distorted_uv).rgb;
        
        // Apply bluish tint to ripples based on normal map intensity
        vec3 tinted_underwater = mix(
            underwater_refracted,  // Original refracted color
            underwater_ripple_color,  // Blue tint color
            ripple_intensity * underwater_ripple_tint  // How much blue to add
        );
        
        // For areas inside Snell's window, use the tinted refraction
        // For areas outside Snell's window, use the water color with blue ripples
        ALBEDO = mix(water_color, tinted_underwater, window_mask);
        
        // Add a slight emission for better underwater visibility
        EMISSION = underwater_ripple_color * ripple_intensity * 0.8;
        
        // Ensure underwater view is fully opaque
        ALPHA = 1.0;
    }
    // Handle surface view (front face)
    else {
        // Calculate fresnel factor for realistic water surface reflectivity
        float fresnel_factor = fresnel(1.0, index_of_refraction, NORMAL, VIEW);
        
        // Blend refracted color with water color based on depth perception
        ALBEDO = mix(refracted_color, water_color, 0.2);
        
        // Apply transparency with fresnel effect (more reflective at grazing angles)
        ALPHA = mix(transparency, 1.0, fresnel_factor * 0.5);
    }
}