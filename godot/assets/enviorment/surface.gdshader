shader_type spatial;
render_mode cull_disabled, shadows_disabled, depth_draw_always;

uniform sampler2D screen_texture : hint_screen_texture, source_color, repeat_disable;
uniform sampler2D normal_map_texture : hint_normal;

// set source color in inspector
uniform vec3 albedo1 : source_color;
uniform vec3 albedo2 : source_color;
uniform float roughness : hint_range(0.0, 1.0) = 0.02;

// Vertex displacement for waves:
uniform sampler2D wave;
uniform float noise_scale = 10.0;
uniform float wave_height_scale = 0.7;
uniform float wave_time_scale = 0.025;
varying float height;
varying vec3 world_position;

// Texture variables for normal map:
uniform sampler2D texture_normal1;
uniform sampler2D texture_normal2;
uniform vec2 direction1 = vec2(2.0, 0.0);
uniform vec2 direction2 = vec2(0.0, 1.0);
uniform float texture_time_scale = 0.005; //: hint_range(0.0, 0.2, 0.005) = 0.02;
uniform float fresnel_size = 4.0;
uniform float fresnel_scale = 2.0;

float fresnel(float amount, vec3 normal, vec3 view) {
	return pow((1.0 - clamp(dot(normalize(normal), normalize(view)), 0.0, 1.0)), amount);
}

// Change vertices to handle waves:
void vertex() {
	world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	// get location of each vertex:
	height = texture(wave, world_position.xz / noise_scale + TIME * wave_time_scale).r;
	VERTEX.y += height * wave_height_scale;
}

void fragment() {
	vec2 time1 = (TIME * direction1) * texture_time_scale;
	vec2 time2 = (TIME * direction2) * texture_time_scale;
	vec3 normal1 = texture(texture_normal1, world_position.xz + time1).rgb; // get noise texture variable and coordinates of noise
	vec3 normal2 = texture(texture_normal2, world_position.xz + time2).rgb;
	vec3 texture_mix = mix(normal1, normal2, 0.5);
	vec3 surface_albedo = mix(albedo1, albedo2, fresnel(fresnel_size, NORMAL, VIEW));
	float surface_alpha = fresnel(fresnel_size, NORMAL, VIEW) * fresnel_scale;

	ALBEDO = surface_albedo;
	ROUGHNESS = roughness;
	NORMAL_MAP = texture_mix;
	ALPHA = surface_alpha;
}