

struct Meshlet
{
	vec4 cone;
	uint32_t vertices[64];
	uint8_t indices[126 * 3];
	uint8_t triangleCount;
	uint8_t vertexCount;
};

struct Vertex
{
	float16_t vx, vy, vz, vw;
	uint8_t nx, ny, nz;
	float16_t tu, tv;
};