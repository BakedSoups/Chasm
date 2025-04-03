shader_type spatial;
render_mode cull_disabled, shadows_disabled, depth_draw_always;

uniform sampler2D screen_texture : hint_screen_texture, source_color, repeat_disable;
uniform sampler2D normal_map_texture : hint_normal;


/* TODO:
[*] remove subsurface and only use surface shader instead
[ ] Snell's window
*/

// Shader Parameters:
// set source color in inspector
uniform vec3 albedo1 : source_color;
uniform vec3 albedo2 : source_color;
//uniform vec4 subsurface_albedo_close = vec4(0.0, 0.0, 0.0, 0.0);
//uniform vec4 subsurface_albedo_far = vec4(0.0, 0.0, 1.0, 1.0);
//uniform float metalic : hint_range(0.0, 1.0) = 0;
uniform float roughness : hint_range(0.0, 1.0) = 0.02;
//uniform float alpha : hint_range(0.0, 1.0, 0.1) = 0.0;

// Vertex displacement:
uniform sampler2D wave;
uniform float noise_scale = 10.0;
uniform float wave_height_scale = 0.7;
uniform float wave_time_scale = 0.025;
varying float height;
varying vec3 world_position;

// Texture Variables:
uniform sampler2D texture_normal1;
uniform sampler2D texture_normal2;
uniform vec2 direction1 = vec2(2.0, 0.0);
uniform vec2 direction2 = vec2(0.0, 1.0);
uniform float texture_time_scale = 0.005; //: hint_range(0.0, 0.2, 0.005) = 0.02;

// fresnel
uniform float fresnel_size = 4.0;
uniform float fresnel_scale = 1.0;


float fresnel(float amount, vec3 normal, vec3 view) {
	return pow((1.0 - clamp(dot(normalize(normal), normalize(view)), 0.0, 1.0)), amount);
}

vec3 normalize_radians(float y, float x) {
	return normalize(vec3(cos(radians(y)) * cos(radians(x)), sin(radians(x)), sin(radians(y)) * cos(radians(x))));  
}

void vertex() {
	// handles waves
	world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	height = texture(wave, world_position.xz / noise_scale + TIME * wave_time_scale).r; // gets r value from noise texture (location of each vertex)
	VERTEX.y += height * wave_height_scale;
}

void fragment() {
	vec2 time1 = (TIME * direction1) * texture_time_scale;
	vec2 time2 = (TIME * direction2) * texture_time_scale;
	vec3 normal1 = texture(texture_normal1, world_position.xz + time1).rgb; // get noise texture variable and coordinates of noise
	vec3 normal2 = texture(texture_normal2, world_position.xz + time2).rgb;
	vec3 normal_blend = mix(normal1, normal2, 0.5);
	vec3 surface_albedo = mix(albedo1, albedo2, fresnel(fresnel_size, NORMAL, VIEW));
	//vec3 subsurface_albedo = mix(albedo1, albedo2, fresnel(fresnel_size, NORMAL, VIEW));
	// Refraction:
	float refracted_angle = dot(normalize(VIEW), normalize(NORMAL)); // the incident angle
	float light_speed = 2.9972458 * pow(10, 8); // constant
	float air_refraction_index = 1.000293; //constant
	float sea_water_refraction_index = 1.39807;  //constant
	//float fresh_water_refraction_index = 1.38919; //constant
	
	float critical_angle = asin(air_refraction_index * sin (90.0) / sea_water_refraction_index);
	
	float air_light_speed = light_speed / air_refraction_index;
	float sea_water_light_speed = light_speed / sea_water_refraction_index;
	
	
	// To reflect or refract that is the quesstion:
	float refracted_light_angle;
	float reflected_light_angle;
	vec3 view_direction;
	/*float light_angle = asin(air_refraction_index * sin(refracted_angle) / sea_water_refraction_index);
	
	refracted_light_angle = asin(air_refraction_index * sin(refracted_angle) / sea_water_refraction_index);
	
	if (refracted_angle < critical_angle) {
		// refract
		view_direction = vec3(refracted_light_angle, NORMAL.z, refracted_angle);
	}
	if (refracted_angle > critical_angle) {
		// reflect
		view_direction = vec3(refracted_light_angle, NORMAL.x, refracted_angle); // kinda works
	}*/
	
	//// refracted_light_angle = asin(air_refraction_index * sin(refracted_angle) / sea_water_refraction_index);


	//vec4 subsurface_albedo = mix(subsurface_albedo_close, subsurface_albedo_far, fresnel(5.0, NORMAL, VIEW));
	//vec3 subsurface_albedo = vec3(surface_albedo.g, surface_albedo.b, 0.0);
	// Fresnel Effect:
	float surface_alpha = fresnel(fresnel_size, NORMAL, VIEW) * fresnel_scale;
	//float refracted_yaw = atan(light_angle);
	//float refracted_pitch = asin();
	//vec3 view_direction = vec3(refracted_light_angle, NORMAL.z, refracted_angle); // kinda works
	float subsurface_alpha; //= fresnel(fresnel_size, NORMAL, VIEW) * fresnel_scale;
	float incident_angle = dot(NORMAL, VIEW);
	
	vec3 albedo = surface_albedo;
	vec4 radiance;
	vec4 irradiance;
	float alpha = surface_alpha;
	float metalic;

	
	if (!FRONT_FACING) { // is below surface
		/* ref: https://godotshaders.com/shader/snells-window/ */
		float time = TIME * 0.01;
		vec3 normal_sample_1 = texture(normal_map_texture, UV + vec2(-1, 1) * time).xyz * 2.0 - 1.0;
		vec3 normal_sample_2 = texture(normal_map_texture, -UV + vec2(0, 1) * time).xyz * 2.0 - 1.0;
		vec3 normal_map = vec3(normal_sample_1.xy + normal_sample_2.xy, 0.0);
		// Set NORMAL directly since we need to use it later
		NORMAL_MAP_DEPTH = 3.0;
		normal_map.z = sqrt(max(0.0, 1.0 - dot(normal_map.xy, normal_map.xy)));
		NORMAL = normalize(mix(NORMAL, TANGENT * normal_map.x + BINORMAL * normal_map.y + NORMAL * normal_map.z, NORMAL_MAP_DEPTH));
		// Screen distortion
		vec2 distorted_uv = SCREEN_UV - normal_map.xy * 2.0 / length(VERTEX);
		ALBEDO = texture(screen_texture, distorted_uv).rgb;
		/* ref end */
		
		//subsurface_alpha = fresnel(fresnel_size, NORMAL, view_direction);
		subsurface_alpha = step(sqrt(1.0 - pow(incident_angle, 2)) * sea_water_refraction_index, 1.0);  // `surface_alpha` determines if water is transparent or reflective
		//albedo = mix(albedo1, ALBEDO, subsurface_alpha);
		//radiance = vec4(vec3(0.0), subsurface_alpha);
		//irradiance = vec4(vec3(1.0), subsurface_alpha);
		//alpha = 1.0;
		//albedo = vec3(0.0, 0.0, 0.5);
		alpha = surface_alpha * fresnel_scale;
	}
	
	ALBEDO = albedo;
	RADIANCE = radiance;
	IRRADIANCE = irradiance;
	METALLIC = metalic;
	ROUGHNESS = roughness;
	NORMAL_MAP = normal_blend;
	ALPHA = alpha; // set closer to 1.0 for deep ocean
	// NORMAL = abs(NORMAL);
}

//void light() {
//	// Called for every pixel for every light affecting the material.
//	// Uncomment to replace the default light processing function with this one.
//}
