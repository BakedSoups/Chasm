shader_type canvas_item;
uniform float scanline_intensity : hint_range(0.0, 1.0) = 0.01;//0.3;
uniform float color_bleed_weight : hint_range(0.0, 1.0) = 0.01;//0.35;
uniform sampler2D screen_texture : hint_screen_texture, filter_nearest;

void fragment() {
	vec2 screen_resolution = vec2(1.0 / SCREEN_PIXEL_SIZE.y, 1.0 / SCREEN_PIXEL_SIZE.x); //vec2(480, 270);
	float color_dark_offset = 0.1;//0.5;
	int y_pos = int(floor(SCREEN_UV.y * screen_resolution.y * 2.0));
	if (int(floor((float(y_pos) / 2.0))) * 2 == y_pos) {
		color_dark_offset = scanline_intensity;
	}
	vec4 adjacent_pixel_color_average = texture(screen_texture, SCREEN_UV - vec2(1.0 / screen_resolution.x, 0)) * 0.5 + texture(screen_texture, SCREEN_UV - vec2(0, 1.0 / screen_resolution.y)) * 0.5;
	vec4 this_pixel_color = texture(screen_texture, SCREEN_UV);
	vec4 screen_color = adjacent_pixel_color_average * color_bleed_weight + this_pixel_color * (1.0 - color_bleed_weight) - vec4(vec3(color_dark_offset), 0);
	vec4 pink = vec4(1.0, 0.0, 0.0, 0.1);
	COLOR = screen_color * pink;
}