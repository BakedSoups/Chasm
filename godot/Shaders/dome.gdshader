// Purpose: Blocks out sun light
shader_type spatial;
uniform vec3 albedo : source_color;
void vertex() {
	// Called for every vertex the material is visible on.
}
void fragment() {
	// Called for every pixel the material is visible on.
	ALBEDO = albedo;
	if (UV.y >= 0.5) {
		ALPHA = 0.0;
	}

}

//void light() {
//	// Called for every pixel for every light affecting the material.
//	// Uncomment to replace the default light processing function with this one.
//}