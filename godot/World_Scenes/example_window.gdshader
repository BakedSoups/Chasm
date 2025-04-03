shader_type spatial;
render_mode cull_disabled;

uniform sampler2D screen_texture : hint_screen_texture, source_color, repeat_disable;
uniform sampler2D normal_map_texture : hint_normal;
uniform vec3 water_color : source_color = vec3(0.0625, 0.1992, 0.3594);
uniform float index_of_refraction = 1.333;

// Returns 1.0 if we can see above the water, 0.0 otherwise
float snells_window(vec3 normal, vec3 view, float ior) {
	float cos_theta = dot(normal, view);
	return step(sqrt(1.0 - cos_theta * cos_theta) * ior, 1.0);
}

void fragment() {
	ROUGHNESS = 0.0;

	/* Basic waves */
	float time = TIME * 0.05;
	vec3 normal_sample_1 = texture(normal_map_texture, UV + vec2(-1, 1) * time).xyz * 2.0 - 1.0;
	vec3 normal_sample_2 = texture(normal_map_texture, -UV + vec2(0, 1) * time).xyz * 2.0 - 1.0;
	vec3 normal_map = vec3(normal_sample_1.xy + normal_sample_2.xy, 0.0);
	
	
	// Set NORMAL directly since we need to use it later
	NORMAL_MAP_DEPTH = 1.0;
	normal_map.z = sqrt(max(0.0, 1.0 - dot(normal_map.xy, normal_map.xy)));
	NORMAL = normalize(mix(NORMAL, TANGENT * normal_map.x + BINORMAL * normal_map.y + NORMAL * normal_map.z, NORMAL_MAP_DEPTH));
	
	/* Screen distortion */
	vec2 distorted_uv = SCREEN_UV - normal_map.xy * 2.0 / length(VERTEX);
	ALBEDO = texture(screen_texture, distorted_uv).rgb;
	
	// Apply Snell's window mask to water surface when viewed from below
	if(!FRONT_FACING) {
		float mask = snells_window(NORMAL, VIEW, index_of_refraction);
		ALBEDO = mix(water_color, ALBEDO, mask);
		RADIANCE = vec4(vec3(0.0), mask);
		IRRADIANCE = vec4(vec3(1.0), mask);
	}
	else {
		ALPHA = 0.5;
	}
}