shader_type spatial;

// Parameters to control the fish animation
uniform float time_scale = 25.0;
uniform float side_to_side = 0.1;   // Side to side motion strength
uniform float pivot = 0.5;          // Pivot motion strength
uniform float wave = 0.1;           // Wave motion strength
uniform float twist = 0.5;          // Twist motion strength
uniform float mask_black = 0.3;     // Start of the motion mask
uniform float mask_white = 0.7;     // End of the motion mask

void vertex() {
    float time = (TIME * time_scale * (0.5 + INSTANCE_CUSTOM.y)) + (6.28318 * INSTANCE_CUSTOM.x);
    
    float body = (VERTEX.z + 1.0) / 2.0;
    
    float mask = smoothstep(mask_black, mask_white, 1.0 - body);
    
    // 1. Side to side motion
    VERTEX.x += cos(time) * side_to_side;
    
    float pivot_angle = cos(time) * 0.1 * pivot;
    mat2 rotation_matrix = mat2(vec2(cos(pivot_angle), -sin(pivot_angle)), 
                              vec2(sin(pivot_angle), cos(pivot_angle)));
    VERTEX.xz = rotation_matrix * VERTEX.xz;
    
    VERTEX.x += cos(time + body) * mask * wave;
    
    float twist_angle = cos(time + body) * 0.3 * twist;
    mat2 twist_matrix = mat2(vec2(cos(twist_angle), -sin(twist_angle)),
                            vec2(sin(twist_angle), cos(twist_angle)));
    VERTEX.xy = mix(VERTEX.xy, twist_matrix * VERTEX.xy, mask);
}

void fragment() {
    ALBEDO = vec3(0.05, 0.05, 0.05);
    
    float highlight = smoothstep(0.7, 1.0, NORMAL.y) * 0.15;
    ALBEDO += vec3(0.0, 0.0, highlight * 2.0);
    
    METALLIC = 0.2;
    ROUGHNESS = 0.7;
}

//void light() {
//    // Called for every pixel for every light affecting the material.
//    // Uncomment to replace the default light processing function with this one.
//}