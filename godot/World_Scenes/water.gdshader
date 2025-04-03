shader_type canvas_item;

// Parameters you can tweak in the Inspector
uniform float water_opacity : hint_range(0.0, 1.0) = 0.7;
uniform float blue_tint_strength : hint_range(0.0, 1.0) = 0.5;
uniform float wave_speed = 1.0;
uniform float wave_frequency = 10.0;
uniform float wave_amplitude = 0.01;
uniform sampler2D SCREEN_TEXTURE : hint_screen_texture, filter_linear_mipmap;

void fragment() {
    // Base blue tint color
    vec3 blue_water_tint = vec3(0.2, 0.4, 0.8);

    // Create simple water movement for light refraction
    vec2 wave_uv = UV;
    wave_uv.x += sin(UV.y * wave_frequency + TIME * wave_speed) * wave_amplitude;
    wave_uv.y += cos(UV.x * wave_frequency + TIME * wave_speed) * wave_amplitude;

    // Sample the screen with the refracted coordinates
    vec4 screen_texture = texture(SCREEN_TEXTURE, SCREEN_UV + (wave_uv - UV) * 0.1);

    // Mix the screen color with the blue tint
    vec3 final_color = mix(screen_texture.rgb, blue_water_tint, blue_tint_strength);

    // Apply water opacity
    COLOR = vec4(final_color, water_opacity);
}