#include "common.h"
#include "shaders.h"



vk::ShaderModule tut::shaders::loadShader(const vk::Device device, const char* path)
{
	// ReSharper disable once CppDeprecatedEntity
	const auto file = fopen(path, "rb");  // NOLINT(clang-diagnostic-deprecated-declarations)
	assert(file);
	fseek(file, 0, SEEK_END);
	const auto length = static_cast<size_t>(ftell(file));
	fseek(file, 0, SEEK_SET);
	const auto buffer = new char[length];

	[[maybe_unused]] auto rc = fread(buffer, 1, length, file);
	fclose(file);


	const auto createInfo = vk::ShaderModuleCreateInfo
	{
		.codeSize = static_cast<size_t>(length),
		.pCode = reinterpret_cast<const U32*>(buffer)

	};
	return returnValueOnSuccess(device.createShaderModule(createInfo));
}


vk::PipelineLayout tut::shaders::createPipelineLayout(const vk::Device device, const bool rtxEnabled)
{
	auto setBindings = std::vector<vk::DescriptorSetLayoutBinding>{};

	if (!rtxEnabled)
	{
		setBindings.push_back(vk::DescriptorSetLayoutBinding
			{
				.binding = 0,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eVertex
			});
	}
	else
	{
		setBindings.push_back(vk::DescriptorSetLayoutBinding
			{
				.binding = 0,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eMeshNV
			});
		setBindings.push_back(vk::DescriptorSetLayoutBinding
			{
				.binding = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eMeshNV
			});

	}



	const auto setLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo
	{
		.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
		.bindingCount = static_cast<U32>(setBindings.size()),
		.pBindings = setBindings.data()
	};

	const auto descriptorSetLayout =
		returnValueOnSuccess(device.createDescriptorSetLayout(setLayoutCreateInfo));

	const auto createInfo = vk::PipelineLayoutCreateInfo
	{
		.setLayoutCount = 1,
		.pSetLayouts = &descriptorSetLayout
	};
	const auto layout = returnValueOnSuccess(device.createPipelineLayout(createInfo));

	//device.destroyDescriptorSetLayout(descriptorSetLayout);

	return layout;
}

vk::Pipeline tut::shaders::createGraphicsPipeline(vk::Device device, vk::PipelineCache pipelineCache, vk::RenderPass renderPass,
	vk::PipelineLayout pipelineLayout, vk::ShaderModule triangleVertexShader,
	vk::ShaderModule triangleFragmentShader, const bool rtxEnabled)
{
	auto stages = std::array
	{
		vk::PipelineShaderStageCreateInfo
		{
			.stage = rtxEnabled ? vk::ShaderStageFlagBits::eMeshNV : vk::ShaderStageFlagBits::eVertex,
			.module = triangleVertexShader,
			.pName = "main"
		},
		vk::PipelineShaderStageCreateInfo
		{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = triangleFragmentShader,
			.pName = "main"
		}
	};

	/*const auto stream = vk::VertexInputBindingDescription
	{
		.binding = 0,
		.stride = 32,
		.inputRate = vk::VertexInputRate::eVertex
	};

	const auto attrs = std::array
	{
		vk::VertexInputAttributeDescription
		{
			.location = 0,
			.binding = 0,
			.format = vk::Format::eR32G32B32Sfloat,
			.offset = 0
		},
		vk::VertexInputAttributeDescription
		{
			.location = 1,
			.binding = 0,
			.format = vk::Format::eR32G32B32Sfloat,
			.offset = 12
		},
		vk::VertexInputAttributeDescription
		{
			.location = 2,
			.binding = 0,
			.format = vk::Format::eR32G32Sfloat,
			.offset = 24
		},

	};
	const auto vertexInput = vk::PipelineVertexInputStateCreateInfo
	{
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &stream,
		.vertexAttributeDescriptionCount = attrs.size(),
		.pVertexAttributeDescriptions = attrs.data()
	};*/
	const auto vertexInput = vk::PipelineVertexInputStateCreateInfo
	{
	};

	auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo
	{
		.topology = vk::PrimitiveTopology::eTriangleList
	};

	auto viewport = vk::PipelineViewportStateCreateInfo
	{
		.viewportCount = 1,
		.scissorCount = 1,
	};

	auto rasterization = vk::PipelineRasterizationStateCreateInfo
	{
		.cullMode = vk::CullModeFlagBits::eBack,
		.lineWidth = 1.0f
	};
	auto multisample = vk::PipelineMultisampleStateCreateInfo
	{
		.rasterizationSamples = vk::SampleCountFlagBits::e1
	};

	auto depthStencil = vk::PipelineDepthStencilStateCreateInfo
	{

	};

	auto colorAttachments = std::array<vk::PipelineColorBlendAttachmentState, 1>
	{
		vk::PipelineColorBlendAttachmentState
		{
			.colorWriteMask = vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
		}
	};

	auto colorBlend = vk::PipelineColorBlendStateCreateInfo
	{
		.attachmentCount = static_cast<U32>(colorAttachments.size()),
		.pAttachments = colorAttachments.data()
	};
	auto dynamicStates = std::array<vk::DynamicState, 2>
	{
		vk::DynamicState::eScissor,
			vk::DynamicState::eViewport
	};
	auto dynamic = vk::PipelineDynamicStateCreateInfo
	{
		.dynamicStateCount = static_cast<U32>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};


	auto createInfo = vk::GraphicsPipelineCreateInfo
	{
		.stageCount = static_cast<U32>(stages.size()),
		.pStages = stages.data(),
		.pVertexInputState = &vertexInput, //fixed function vertex input
		.pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlend,
		.pDynamicState = &dynamic,
		.layout = pipelineLayout,
		.renderPass = renderPass
	};
	return returnValueOnSuccess(device.createGraphicsPipeline(pipelineCache, createInfo));
}
