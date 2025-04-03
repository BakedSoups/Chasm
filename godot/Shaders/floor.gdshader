shader_type spatial;

void vertex() {
	// Called for every vertex the material is visible on.

}

void fragment() {
	// Called for every pixel the material is visible on.
	ALBEDO = vec3(0.194, 0.178, 0.128);
}

//void light() {
//	// Called for every pixel for every light affecting the material.
//	// Uncomment to replace the default light processing function with this one.
//}
