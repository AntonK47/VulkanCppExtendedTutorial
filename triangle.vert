#version 450
//#extension GL_KHR_vulkan_glsl: enable

//const vec3 verticies[] =
//{
//	vec3(0.0,0.5,0.0),
//	vec3(0.5,-0.5,0.0),
//	vec3(-0.5,-0.5,0.0)
//};
//
//
//void main()
//{
//	gl_Position = vec4(verticies[gl_VertexIndex], 1.0);
//}
layout(location = 0)
in vec3 position;
layout(location = 1)
in vec3 normal;
layout(location = 2)
in vec3 texCoord;

layout(location = 0)
out vec4 color;

void main()
{
	gl_Position = vec4(position + vec3(0,0,0.5), 1.0);
	color = vec4(normal * 0.5 + vec3(0.5), 1.0);
}