#pragma once


namespace tut::shaders
{
	vk::ShaderModule loadShader(const vk::Device device, const char* path);
	vk::PipelineLayout createPipelineLayout(const vk::Device device, const bool rtxEnabled);
	vk::Pipeline createGraphicsPipeline(vk::Device device, vk::PipelineCache pipelineCache, vk::RenderPass renderPass,
	vk::PipelineLayout pipelineLayout, vk::ShaderModule triangleVertexShader,
	vk::ShaderModule triangleFragmentShader, const bool rtxEnabled);

	struct DescriptorInfo
	{
		union
		{
			vk::DescriptorImageInfo image;
			vk::DescriptorBufferInfo buffer;
		};
	};

}
