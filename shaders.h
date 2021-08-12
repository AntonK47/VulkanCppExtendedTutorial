#pragma once

#include "spirv_reflect.h"
namespace tut::shaders
{
	struct Shader
	{
		vk::ShaderModule module;
		vk::ShaderStageFlagBits stage;
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
	};

	Shader loadShader(const vk::Device device, const char* path);
	vk::DescriptorSetLayout createSetLayout(const vk::Device device, std::initializer_list<std::reference_wrapper<Shader>> shaders);
	vk::PipelineLayout createPipelineLayout(const vk::Device device, const vk::DescriptorSetLayout setLayout);
	vk::DescriptorUpdateTemplate createUpdateTemplate(const vk::Device device, const vk::PipelineBindPoint bindPoint,
	                                                  const vk::DescriptorSetLayout setLayout,
	                                                  const vk::PipelineLayout layout, std::initializer_list<std::reference_wrapper<Shader>> shaders);
	vk::Pipeline createGraphicsPipeline(vk::Device device, vk::PipelineCache pipelineCache, vk::RenderPass renderPass,
	                                    vk::PipelineLayout pipelineLayout, std::initializer_list<std::reference_wrapper<Shader>> shaders);

	struct DescriptorInfo
	{
		union
		{
			vk::DescriptorImageInfo image;
			vk::DescriptorBufferInfo buffer;
		};
	};
	
}
