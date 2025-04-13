shader_type canvas_item;

uniform sampler2D SCREEN_TEXTURE: hint_screen_texture, filter_linear_mipmap;
uniform int pixel_size : hint_range(1, 64) = 4; // Pixel size

void fragment() {
    // Get the size of the screen texture (dynamic screen resolution)
    vec2 screen_size = 1.0 / SCREEN_PIXEL_SIZE;
    
    // Pixel coordinates in screen space
    vec2 pixel_coords = floor(FRAGCOORD.xy / float(pixel_size)) * float(pixel_size);
    
    // Convert pixel coordinates to UVs for screen texture
    vec2 uv = pixel_coords / screen_size;
    
    // Get color from screen texture
    COLOR = texture(SCREEN_TEXTURE, uv);
}