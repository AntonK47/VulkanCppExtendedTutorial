#include "common.h"
#include "shaders.h"


tut::shaders::Shader tut::shaders::loadShader(const vk::Device device, const char* path)
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


	const auto reflect = spv_reflect::ShaderModule{static_cast<size_t>(length),reinterpret_cast<const U32*>(buffer)};
	const auto stage = static_cast<vk::ShaderStageFlagBits>(reflect.GetShaderStage());

	auto setLayoutBindings = std::vector<vk::DescriptorSetLayoutBinding>{};
	auto bindingsCount = U32{};
	reflect.EnumerateDescriptorBindings(&bindingsCount, nullptr);

	auto descriptorBindings = std::vector<SpvReflectDescriptorBinding*>{bindingsCount};
	reflect.EnumerateDescriptorBindings(&bindingsCount, descriptorBindings.data());

	for (const auto& descriptorBinding : descriptorBindings)
	{
		setLayoutBindings.push_back(
			vk::DescriptorSetLayoutBinding
			{
				.binding = descriptorBinding->binding,
				.descriptorType = vk::DescriptorType{descriptorBinding->descriptor_type},
				.descriptorCount = descriptorBinding->count,
				.stageFlags = stage
			});
	}

	return Shader
	{
		.module = returnValueOnSuccess(device.createShaderModule(createInfo)),
		.stage = stage,
		.setLayoutBindings = setLayoutBindings
	};
}

vk::DescriptorSetLayout tut::shaders::createSetLayout(const vk::Device device, std::initializer_list<std::reference_wrapper<Shader>> shaders)
{
	auto setBindings = std::vector<vk::DescriptorSetLayoutBinding>{};
	for (const auto& shader : shaders)
	{
		setBindings.insert(setBindings.end(), shader.get().setLayoutBindings.begin(), shader.get().setLayoutBindings.end());
	}
	const auto setLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo
	{
		.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
		.bindingCount = static_cast<U32>(setBindings.size()),
		.pBindings = setBindings.data()
	};

	return returnValueOnSuccess(device.createDescriptorSetLayout(setLayoutCreateInfo));
}

vk::PipelineLayout tut::shaders::createPipelineLayout(const vk::Device device, const vk::DescriptorSetLayout setLayout)
{
	
	const auto createInfo = vk::PipelineLayoutCreateInfo
	{
		.setLayoutCount = 1,
		.pSetLayouts = &setLayout
	};
	const auto layout = returnValueOnSuccess(device.createPipelineLayout(createInfo));

	return layout;
}

vk::DescriptorUpdateTemplate tut::shaders::createUpdateTemplate(const vk::Device device, const vk::PipelineBindPoint bindPoint,
	const vk::DescriptorSetLayout setLayout, const vk::PipelineLayout layout, std::initializer_list<std::reference_wrapper<Shader>> shaders)
{

	auto setBindings = std::vector<vk::DescriptorSetLayoutBinding>{};
	for (const auto& shader : shaders)
	{
		setBindings.insert(setBindings.end(), shader.get().setLayoutBindings.begin(), shader.get().setLayoutBindings.end());
	}
	
	auto entries = std::vector<vk::DescriptorUpdateTemplateEntry>{};


	for (const auto& binding : setBindings)
	{
		entries.push_back(
			vk::DescriptorUpdateTemplateEntry
			{
				binding.binding,
				0,
				binding.descriptorCount,
				vk::DescriptorType::eStorageBuffer,
				sizeof(tut::shaders::DescriptorInfo) * binding.binding,
				sizeof(tut::shaders::DescriptorInfo)
			}
		);
	}

	const auto createInfo = vk::DescriptorUpdateTemplateCreateInfo
	{
		.descriptorUpdateEntryCount = static_cast<U32>(entries.size()),
		.pDescriptorUpdateEntries = entries.data(),
		.templateType = vk::DescriptorUpdateTemplateType::ePushDescriptorsKHR,
		.descriptorSetLayout = setLayout,
		.pipelineBindPoint = bindPoint,
		.pipelineLayout = layout,
		.set = 0
	};
	return returnValueOnSuccess(device.createDescriptorUpdateTemplate(createInfo));
}

vk::Pipeline tut::shaders::createGraphicsPipeline(vk::Device device, vk::PipelineCache pipelineCache, vk::RenderPass renderPass,
                                                  vk::PipelineLayout pipelineLayout, std::initializer_list<std::reference_wrapper<Shader>> shaders)
{
	

	auto stages = std::vector<vk::PipelineShaderStageCreateInfo>{};
	for (auto shader : shaders)
	{
		stages.push_back(
			vk::PipelineShaderStageCreateInfo
			{
				.stage = shader.get().stage,
				.module = shader.get().module,
				.pName = "main"
			}
		);
	};

	
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
		.cullMode = vk::CullModeFlagBits::eFront,
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
