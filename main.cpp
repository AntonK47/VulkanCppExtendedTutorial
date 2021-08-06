
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR

#define WIN32_LEAN_AND_MEAN
// ReSharper disable once IdentifierTypo
#define NOMINMAX

#include <Windows.h>
#endif



// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS


#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1  // NOLINT(cppcoreguidelines-macro-usage)

#include <compare>

#pragma warning(push)
#pragma warning( disable : 26819 )
#pragma warning( disable : 26812 )
#pragma warning( disable : 28251 )
#pragma warning( disable : 26812 )
#pragma warning( disable : 26495 )
#pragma warning( disable : 4464 )
#pragma warning( disable : 4820 )
#include <boost/assert.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.hpp>

#define FAST_OBJ_IMPLEMENTATION
#include <fast_obj.h>
#include <meshoptimizer.h>

#pragma warning(pop)


#include <iostream>
#include <iomanip>
#include <vector>


#define RTX  1// NOLINT(cppcoreguidelines-macro-usage)

using U32 = uint32_t;

template<typename T>
T returnValueOnSuccess(const vk::ResultValue<T>& call) { assert(call.result == vk::Result::eSuccess); return call.value; }


void returnValueOnSuccess(const vk::Result& result) { assert(result == vk::Result::eSuccess); }


void setupConsole(const std::wstring& title)
{
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	FILE* stream;
	freopen_s(&stream, "CONIN$", "r", stdin);
	freopen_s(&stream, "CONOUT$", "w+", stdout);
	freopen_s(&stream, "CONOUT$", "w+", stderr);
	SetConsoleTitle(title.c_str());
}

bool supportPresentation(const vk::PhysicalDevice physicalDevice, const U32 familyIndex)
{
	return physicalDevice.getWin32PresentationSupportKHR(familyIndex);
}

U32 getGraphicsQueueFamilyIndex(const vk::PhysicalDevice physicalDevice)
{
	const auto properties = physicalDevice.getQueueFamilyProperties();
	for (U32 i = 0; i < properties.size(); i++)
	{
		if (properties[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			return i;
		}
	}
	return VK_QUEUE_FAMILY_IGNORED;
}

vk::PhysicalDevice pickPhysicalDevice(const std::vector<vk::PhysicalDevice>& physicalDevices)
{
	auto discrete = vk::PhysicalDevice{ nullptr };
	auto fallback = vk::PhysicalDevice{ nullptr };

	
	BOOST_ASSERT(!physicalDevices.empty());
	for (const auto physicalDevice : physicalDevices)
	{
		const auto familyIndex = getGraphicsQueueFamilyIndex(physicalDevice);
		if(familyIndex == VK_QUEUE_FAMILY_IGNORED)
		{
			continue;
		}
		if(!supportPresentation(physicalDevice, familyIndex))
		{
			continue;
		}
		if (!discrete)
		{
			discrete = physicalDevice;
		}
		if(!fallback)
		{
			fallback = physicalDevice;
		}
	}
	const auto result = discrete ? discrete : fallback;
	if(result)
	{
		const auto props = result.getProperties();
		std::cout << "Picking discrete GPU: " << props.deviceName << "\n";
	}
	else
	{
		std::cout << "ERROR: Np GPUs found!\n";
	}
	return result;
}

vk::Device createDevice(const vk::PhysicalDevice physicalDevice, U32 familyIndex)
{
	auto queuePriorities = std::array<float, 1>{ 1.0f };
	auto deviceQueueCreateInfo = vk::DeviceQueueCreateInfo
	{
		.queueFamilyIndex = familyIndex,
		.queueCount = 1,
		.pQueuePriorities = queuePriorities.data()
	};


	auto extensions = std::array
	{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
		VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
#if RTX
		VK_NV_MESH_SHADER_EXTENSION_NAME
#endif
	};
	

	const auto features = vk::StructureChain<vk::PhysicalDeviceFeatures2,
		vk::PhysicalDevice16BitStorageFeatures,
		vk::PhysicalDevice8BitStorageFeatures,
		vk::PhysicalDeviceShaderFloat16Int8Features
#if RTX
		,vk::PhysicalDeviceMeshShaderFeaturesNV
#endif

	>
	{
		vk::PhysicalDeviceFeatures2
		{
			.features = vk::PhysicalDeviceFeatures{}
		},
		vk::PhysicalDevice16BitStorageFeatures
		{
			.storageBuffer16BitAccess = vk::Bool32{true},
			.uniformAndStorageBuffer16BitAccess = vk::Bool32{true}
		},
		vk::PhysicalDevice8BitStorageFeatures
		{
			.storageBuffer8BitAccess = vk::Bool32{true},
			.uniformAndStorageBuffer8BitAccess = vk::Bool32{true}
		},
		vk::PhysicalDeviceShaderFloat16Int8Features
		{
			.shaderFloat16 = vk::Bool32{true},
			.shaderInt8 = vk::Bool32{true}
		},
#if RTX
		vk::PhysicalDeviceMeshShaderFeaturesNV
		{
			.taskShader = vk::Bool32{true},
			.meshShader = vk::Bool32{true}
		}
#endif

	};

	const auto deviceCreateInfo = vk::DeviceCreateInfo
	{
		.pNext = &features.get(),
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<U32>(extensions.size()),
		.ppEnabledExtensionNames = extensions.data(),
		
	};
	return returnValueOnSuccess(physicalDevice.createDevice(deviceCreateInfo));
}

vk::Format getSwapchainFormat(const vk::PhysicalDevice physicalDevice, const vk::SurfaceKHR surface)
{
	const auto formats = returnValueOnSuccess(physicalDevice.getSurfaceFormatsKHR(surface));

	if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined)
	{
		return vk::Format::eR8G8B8A8Unorm;
	}

	for (const auto& [format, colorSpace] : formats)
	{
		if (format == vk::Format::eA2R10G10B10UnormPack32)
		{
			return format;
		}
	}

	return formats[0].format;
}

vk::SwapchainKHR createSwapchain(const vk::Device device, vk::SurfaceKHR surface,
                                 vk::SurfaceCapabilitiesKHR surfaceCapabilities, U32 familyIndex, U32 width, U32 height,
                                 vk::Format format, vk::SwapchainKHR oldSwapchain)
{
	
	auto supportedCompositeAlpha =
		surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque ? vk::CompositeAlphaFlagBitsKHR::eOpaque :
		surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
		? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
		:
		surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
		? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
		:
		vk::CompositeAlphaFlagBitsKHR::eInherit;

	const auto swapchainCreateInfo = vk::SwapchainCreateInfoKHR
	{
		.surface = surface,
		.minImageCount = std::clamp(2u, surfaceCapabilities.minImageCount, surfaceCapabilities.maxImageCount),
		.imageFormat = format,
		.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear, //!
		.imageExtent =
			vk::Extent2D
			{
				.width = width,
				.height = height
			},
		.imageArrayLayers = 1,
		.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = std::array<U32,1>{ familyIndex }.data(),
		.preTransform = surfaceCapabilities.currentTransform,
		.compositeAlpha = supportedCompositeAlpha,
		.presentMode = vk::PresentModeKHR::eFifo,
		.oldSwapchain = oldSwapchain
	};
	return returnValueOnSuccess(device.createSwapchainKHR(swapchainCreateInfo));
}

vk::Semaphore createSemaphore(const vk::Device device)
{
	return returnValueOnSuccess(device.createSemaphore({}));
}

vk::RenderPass createRenderPass(const vk::Device device, vk::Format format)
{
	auto attachmentReference = vk::AttachmentReference
	{
		.attachment = 0,
		.layout = vk::ImageLayout::eColorAttachmentOptimal
	};

	auto subpass = vk::SubpassDescription
	{
		.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentReference
	};

	auto attachment = vk::AttachmentDescription
	{
		.format = format,
		.samples = vk::SampleCountFlagBits::e1,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout = vk::ImageLayout::eUndefined,
		.finalLayout = vk::ImageLayout::ePresentSrcKHR
	};

	const auto createInfo = vk::RenderPassCreateInfo
	{
		.attachmentCount = 1,
		.pAttachments = &attachment,
		.subpassCount = 1,
		.pSubpasses = &subpass
	};
	return returnValueOnSuccess(device.createRenderPass(createInfo));
}

vk::Framebuffer createFramebuffer(const vk::Device device, vk::RenderPass renderPass, vk::ImageView imageView,
                                  U32 width, U32 height)
{
	const auto createInfo = vk::FramebufferCreateInfo
	{
		.renderPass = renderPass,
		.attachmentCount = 1,
		.pAttachments = &imageView,
		.width = width,
		.height = height,
		.layers = 1
	};
	return returnValueOnSuccess(device.createFramebuffer(createInfo));
}

vk::ImageView createImageView(const vk::Device device, vk::Image image, vk::Format format)
{
	const auto createInfo = vk::ImageViewCreateInfo
	{
		.image = image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
		.subresourceRange = vk::ImageSubresourceRange
		{ 
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.levelCount = 1,
			.layerCount = 1
		}
	};
	return returnValueOnSuccess(device.createImageView(createInfo));
}





vk::ShaderModule loadShader(const vk::Device device, const char* path)
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

vk::PipelineLayout createPipelineLayout(const vk::Device device)
{
#if RTX
	const auto setBindings = std::array
	{
		vk::DescriptorSetLayoutBinding
		{
			.binding = 0,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eMeshNV
		},
		vk::DescriptorSetLayoutBinding
		{
			.binding = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eMeshNV
		}
	};
#else
	const auto setBindings = std::array{ vk::DescriptorSetLayoutBinding
		{
			.binding = 0,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eVertex
		}
	};
#endif

	

	const auto setLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo
	{
		.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
		.bindingCount = setBindings.size(),
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

vk::Pipeline createGraphicsPipeline(vk::Device device, vk::PipelineCache pipelineCache, vk::RenderPass renderPass,
                                    vk::PipelineLayout pipelineLayout, vk::ShaderModule triangleVertexShader,
                                    vk::ShaderModule triangleFragmentShader)
{
	auto stages = std::array
	{
#if RTX
		vk::PipelineShaderStageCreateInfo
		{
			.stage = vk::ShaderStageFlagBits::eMeshNV,
			.module = triangleVertexShader,
			.pName = "main"
		},
#else
		vk::PipelineShaderStageCreateInfo
		{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = triangleVertexShader,
			.pName = "main"
		},
#endif

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


VkBool32 debugReportCallback(VkDebugReportFlagsEXT flags, [[maybe_unused]] VkDebugReportObjectTypeEXT objectType,
                             [[maybe_unused]] uint64_t object, [[maybe_unused]] size_t location,
                             [[maybe_unused]] int32_t messageCode, [[maybe_unused]] const char* pLayerPrefix,
                             const char* pMessage, [[maybe_unused]] void* pUserData)
{
	
	const auto level = vk::to_string(flags & static_cast<U32>(vk::DebugReportFlagBitsEXT::eError)
		                                 ? vk::DebugReportFlagBitsEXT::eError
		                                 : flags & (static_cast<U32>(vk::DebugReportFlagBitsEXT::eWarning) | static_cast
			                                 <U32>(vk::DebugReportFlagBitsEXT::ePerformanceWarning))
		                                 ? vk::DebugReportFlagBitsEXT::eWarning
		                                 : vk::DebugReportFlagBitsEXT::eInformation);

	std::cout << "[" << level << "]: " << pMessage << std::endl;

	//BOOST_ASSERT_MSG(flags & static_cast<U32>(vk::DebugReportFlagBitsEXT::eError), L"Validation error encountered!");
	return VK_FALSE;
}

vk::DebugReportCallbackEXT registerDebugCallback(const vk::Instance instance)
{
	const auto createInfo = vk::DebugReportCallbackCreateInfoEXT
	{
		.flags = vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning |
		vk::DebugReportFlagBitsEXT::ePerformanceWarning | vk::DebugReportFlagBitsEXT::eInformation,
		.pfnCallback = debugReportCallback
	};

	return returnValueOnSuccess(instance.createDebugReportCallbackEXT(createInfo));
}

vk::ImageMemoryBarrier imageBarrier(vk::Image image,
	vk::AccessFlags srcAccessMask,
	vk::AccessFlags dstAccessMask,
	vk::ImageLayout oldLayout,
	vk::ImageLayout newLayout)
{
	return vk::ImageMemoryBarrier
	{
		.srcAccessMask = srcAccessMask,
		.dstAccessMask = dstAccessMask,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange =
		vk::ImageSubresourceRange
		{
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.levelCount = VK_REMAINING_MIP_LEVELS,
			.layerCount = VK_REMAINING_ARRAY_LAYERS
		}
	};
}

struct Swapchain
{
	vk::SwapchainKHR swapchain;
	std::vector<vk::Image> images;
	std::vector<vk::Framebuffer> framebuffers;
	std::vector<vk::ImageView> imageViews;

	U32 width;
	U32 height;
};

void destroySwapchain(const vk::Device device, const Swapchain& swapchain)
{
	for (const auto& framebuffer : swapchain.framebuffers)
	{
		device.destroy(framebuffer);
	}
	for (const auto& imageView : swapchain.imageViews)
	{
		device.destroyImageView(imageView);
	}
	device.destroySwapchainKHR(swapchain.swapchain);
}

Swapchain createSwapchain(const vk::Device device, const vk::SurfaceKHR surface,
                          const vk::SurfaceCapabilitiesKHR surfaceCapabilities, const U32 familyIndex, const U32 width,
                          const U32 height, const vk::Format format, const vk::RenderPass renderPass,
                          const vk::SwapchainKHR oldSwapchain)
{
	auto swapchain = createSwapchain(device, surface, surfaceCapabilities, familyIndex, width,
	                                 height, format, oldSwapchain);
	auto images = returnValueOnSuccess(device.getSwapchainImagesKHR(swapchain));
	auto views = std::vector<vk::ImageView>{ images.size() };
	auto framebuffers = std::vector<vk::Framebuffer>{ images.size() };
	
	for (U32 i = 0; i < views.size(); i++)
	{
		views[i] = createImageView(device, images[i], format);
	}

	for (U32 i = 0; i < framebuffers.size(); i++)
	{
		framebuffers[i] = createFramebuffer(device, renderPass, views[i], width, height);
	}
	
	return Swapchain
	{
		.swapchain = swapchain,
		.images = images,
		.framebuffers = framebuffers,
		.imageViews = views,
		.width = width,
		.height = height
	};
}

void resizeSwapchain(const vk::Device device, const vk::SurfaceKHR surface,
                     const vk::SurfaceCapabilitiesKHR surfaceCapabilities, const U32 familyIndex, const U32 width,
                     const U32 height, const vk::Format format, const vk::RenderPass renderPass,
                     Swapchain& swapchain)
{
	const auto oldSwapchain = swapchain;

	const auto newSwapchain = createSwapchain(device, surface, surfaceCapabilities, familyIndex, width, height, format, renderPass, swapchain.swapchain);
	returnValueOnSuccess(device.waitIdle());
	destroySwapchain(device, oldSwapchain);
	swapchain = newSwapchain;
}


struct Vertex
{
	uint16_t vx, vy, vz, vw;
	uint8_t nx, ny, nz;
	uint16_t tu, tv;
};

struct Meshlet
{
	uint32_t vertices[64];
	uint8_t indices[126*3];
	uint8_t triangleCount;
	uint8_t vertexCount;
};

struct Mesh
{
	std::vector<Vertex> vertices;
	std::vector<U32> indices;
	std::vector<Meshlet> meshlets;
};

Mesh loadMesh(const char* path)
{
	const auto meshData = fast_obj_read(path);


	U32 totalIndices = 0;

	for (U32 i = 0; i < meshData->face_count; ++i)
	{
		totalIndices += (meshData->face_vertices[i] - 2) * 3;
	}
	auto vertices = std::vector<Vertex>{ totalIndices };

	U32 vertexOffset = 0;
	U32 indexOffset = 0;

	for (U32 i = 0; i < meshData->face_count; ++i)
	{
		for (U32 j = 0; j < meshData->face_vertices[i]; ++j)
		{
			const auto [p, t, n] = meshData->indices[indexOffset + j];
			const auto nx = meshData->normals[n * 3 + 0];
			const auto ny = meshData->normals[n * 3 + 1];
			const auto nz = meshData->normals[n * 3 + 2];
			const Vertex v =
			{
				meshopt_quantizeHalf( meshData->positions[p * 3 + 0]),
				meshopt_quantizeHalf(meshData->positions[p * 3 + 1]),
				meshopt_quantizeHalf(meshData->positions[p * 3 + 2]),
				static_cast<uint8_t>(nx * 127.0f + 127.0f),
				static_cast<uint8_t>(ny * 127.0f + 127.0f),
				static_cast<uint8_t>(nz * 127.0f + 127.0f),
				meshopt_quantizeHalf(meshData->texcoords[t * 2 + 0]),
				meshopt_quantizeHalf(meshData->texcoords[t * 2 + 1]),
			};

			// triangulate polygon on the fly; offset-3 is always the first polygon vertex
			if (j >= 3)
			{
				vertices[vertexOffset + 0] = vertices[vertexOffset - 3];
				vertices[vertexOffset + 1] = vertices[vertexOffset - 1];
				vertexOffset += 2;
			}

			vertices[vertexOffset] = v;
			vertexOffset++;
		}

		indexOffset += meshData->face_vertices[i];
	}

	fast_obj_destroy(meshData);

	std::vector<U32> remap(totalIndices);

	const auto totalVertices = meshopt_generateVertexRemap(&remap[0], nullptr, totalIndices, &vertices[0],
	                                                          totalIndices, sizeof(Vertex));

	auto result = Mesh{};
	
	result.indices.resize(totalIndices);
	meshopt_remapIndexBuffer(&result.indices[0], nullptr, totalIndices, &remap[0]);

	result.vertices.resize(totalVertices);
	meshopt_remapVertexBuffer(&result.vertices[0], &vertices[0], totalIndices, sizeof(Vertex), &remap[0]);

	meshopt_optimizeVertexCache(result.indices.data(), result.indices.data(), result.indices.size(), result.vertices.size());
	meshopt_optimizeVertexFetch(result.vertices.data(), result.indices.data(), result.indices.size(), result.vertices.data(), result.vertices.size(), sizeof(Vertex));


	return result;
}

struct Buffer
{
	vk::Buffer buffer;
	vk::DeviceMemory memory;
	void* data{};// std::unique_ptr<void*> data;
	U32 size{};
};

U32 selectMemoryType(const vk::PhysicalDeviceMemoryProperties& memoryProperties, U32 memoryTypeBits, const vk::MemoryPropertyFlags flags)
{
	U32 result = -1;
	for(U32 i = 0; memoryProperties.memoryTypeCount; i++)
	{
		if((memoryTypeBits & 1 << i) != 0 && (memoryProperties.memoryTypes[i].propertyFlags & flags) == flags)
		{
			result = i;
			break;
		}
	}

	BOOST_ASSERT_MSG(result >= static_cast<U32>(0), "No compatible memory type found!");
	return result;
}

Buffer createBuffer(const vk::Device device, const vk::PhysicalDeviceMemoryProperties& memoryProperties, const U32 size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags memoryPropertyFlags)
{

	const auto createInfo = vk::BufferCreateInfo
	{
		.size = size,
		.usage = usage
	};

	const auto buffer = returnValueOnSuccess(device.createBuffer(createInfo));

	const auto memoryRequirements = device.getBufferMemoryRequirements(buffer);

	auto memoryHeapIndex = selectMemoryType(memoryProperties, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	
	const auto allocateInfo = vk::MemoryAllocateInfo
	{
		.allocationSize = memoryRequirements.size,
		.memoryTypeIndex = memoryHeapIndex
	};
	auto memory = returnValueOnSuccess(device.allocateMemory(allocateInfo));

	returnValueOnSuccess(device.bindBufferMemory(buffer, memory, 0));

	if(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)
	{
		auto dataPtr = returnValueOnSuccess(device.mapMemory(memory, 0, memoryRequirements.size));
		return Buffer
		{
			.buffer = buffer,
			.memory = memory,
			.data = dataPtr,// std::make_unique<void*>(dataPtr),
			.size = static_cast<U32>(memoryRequirements.size)
		};
	}




	return Buffer
	{
		.buffer = buffer,
		.memory = memory,
		.data = nullptr,// dataPtr,// std::make_unique<void*>(dataPtr),
		.size = static_cast<U32>(memoryRequirements.size)
	};
}

template <class T>
void uploadBuffer(vk::Device device, const vk::CommandPool commandPool,
	const vk::CommandBuffer commandBuffer, 
	const vk::Queue queue, const Buffer& srcBuffer,
	const Buffer& dstBuffer, std::vector<T> data)
{
	const auto size = data.size() * sizeof(T);
	memcpy(srcBuffer.data, data.data(), size);

	const auto fullBufferRegion = std::array{
		vk::BufferCopy
		{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = vk::DeviceSize{size}
		}
	};
	




	returnValueOnSuccess(device.resetCommandPool(commandPool));

	const auto beginInfo = vk::CommandBufferBeginInfo
	{
		.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	};
	returnValueOnSuccess(commandBuffer.begin(beginInfo));

	commandBuffer.copyBuffer(srcBuffer.buffer, dstBuffer.buffer, fullBufferRegion);

	const auto bufferBarrier = vk::BufferMemoryBarrier
	{
		.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
		.dstAccessMask = vk::AccessFlagBits::eShaderRead,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = dstBuffer.buffer,
		.offset = 0,
		.size = size
	};
	
	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion, nullptr,
		std::array{ bufferBarrier }, nullptr);

	returnValueOnSuccess(commandBuffer.end());

	
	const auto submits = std::array
	{
		vk::SubmitInfo
		{
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffer
		}
	};
	returnValueOnSuccess(queue.submit(submits));

	returnValueOnSuccess(device.waitIdle());

}

void destroyBuffer(const Buffer& buffer, const vk::Device device)
{
	device.freeMemory(buffer.memory);
	device.destroyBuffer(buffer.buffer);
}

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE


void buildMeshlets(Mesh& mesh)
{
	auto meshlet = Meshlet{};

	constexpr auto invalidVertexValue = std::numeric_limits<uint8_t>::max();
	auto meshletVertices = std::vector( mesh.vertices.size(), invalidVertexValue);

	for (U32 i = 0; i < static_cast<U32>(mesh.indices.size()); i+=3)
	{
		const auto a = mesh.indices[i + 0];
		const auto b = mesh.indices[i + 1];
		const auto c = mesh.indices[i + 2];

		auto& av = meshletVertices[a];
		auto& bv = meshletVertices[b];
		auto& cv = meshletVertices[c];

		const auto newVertexCount = meshlet.vertexCount + (av == invalidVertexValue) + (bv == invalidVertexValue) + (cv == invalidVertexValue);

		if(newVertexCount > 64 || meshlet.triangleCount >= 126)
		{
			mesh.meshlets.push_back(meshlet);
			meshlet = {};
			memset(meshletVertices.data(), invalidVertexValue, mesh.vertices.size());
		}

		if(av == invalidVertexValue)
		{
			av = meshlet.vertexCount;
			meshlet.vertices[meshlet.vertexCount++] = a;
		}
		if(bv == invalidVertexValue)
		{
			bv = meshlet.vertexCount;
			meshlet.vertices[meshlet.vertexCount++] = b;
		}
		if(cv == invalidVertexValue)
		{
			cv = meshlet.vertexCount;
			meshlet.vertices[meshlet.vertexCount++] = c;
		}

		meshlet.indices[meshlet.triangleCount * 3 + 0] = av;
		meshlet.indices[meshlet.triangleCount * 3 + 1] = bv;
		meshlet.indices[meshlet.triangleCount * 3 + 2] = cv;
		meshlet.triangleCount++;
	}

	if(meshlet.triangleCount)
	{
		mesh.meshlets.push_back(meshlet);
	}
}


vk::QueryPool createQueryPool(const vk::Device& device, U32 poolCount)
{
	const auto createInfo = vk::QueryPoolCreateInfo
	{
		.queryType = vk::QueryType::eTimestamp,
		.queryCount = poolCount
	};
	return returnValueOnSuccess(device.createQueryPool(createInfo));
}

int main()  // NOLINT(bugprone-exception-escape)
{



	
	vk::DynamicLoader dl;
	auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

	setupConsole(L"Vulkan Console");

	// Create an SDL window that supports Vulkan rendering.
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
	{
		std::cout << "Could not initialize SDL." << std::endl;
		return 1;
	}
	auto window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
	if (window == nullptr)
	{
		std::cout << "Could not create SDL window." << std::endl;
		return 1;
	}

	// Get WSI extensions from SDL (we can add more if we like - we just can't remove these)
	unsigned extensionCount;
	if (!SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr)) {
		std::cout << "Could not get the number of required instance extensions from SDL." << std::endl;
		return 1;
	}
	auto extensions = std::vector<const char*>{ extensionCount };
	if (!SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data()))
	{
		std::cout << "Could not get the names of required instance extensions from SDL." << std::endl;
		return 1;
	}

	//make possible to use report callback
	extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

	// Use validation layers if this is a debug build
	auto layers = std::vector<const char*>{};
#if defined(_DEBUG)
	layers.push_back("VK_LAYER_KHRONOS_validation");
	//layers.push_back("VK_LAYER_NV_nomad_release_public_2021_3_1");
#endif

	// vk::ApplicationInfo allows the programmer to specify some basic information about the
	// program, which can be useful for layers and tools to provide more debug information.
	auto appInfo = vk::ApplicationInfo
	{
		.pApplicationName = "Vulkan C++ Windowed Program Template",
		.applicationVersion = 1,
		.pEngineName = "LunarG SDK",
		.engineVersion = 1,
		.apiVersion = VK_API_VERSION_1_2
	};


	// vk::InstanceCreateInfo is where the programmer specifies the layers and/or extensions that
	// are needed.
	auto instInfo = vk::InstanceCreateInfo
	{
		.flags = vk::InstanceCreateFlags(),
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<U32>(layers.size()),
		.ppEnabledLayerNames = layers.data(),
		.enabledExtensionCount = static_cast<U32>(extensions.size()),
		.ppEnabledExtensionNames = extensions.data(),
	};

	// Create the Vulkan instance.
	auto resultValue = vk::createInstance(instInfo);
	if (resultValue.result != vk::Result::eSuccess)
	{
		std::cout << "Could not create a Vulkan instance." << std::endl;
		return 1;
	}
	auto instance = resultValue.value;
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

	auto debugCallback = registerDebugCallback(instance);



	// Create a Vulkan surface for rendering
	auto surfaceCType = vk::SurfaceKHR::CType{};
	if (!SDL_Vulkan_CreateSurface(window, instance, &surfaceCType))
	{
		std::cout << "Could not create a Vulkan surface." << std::endl;
		return 1;
	}
	auto surface = vk::SurfaceKHR{ surfaceCType };

	auto physicalDevices = returnValueOnSuccess(instance.enumeratePhysicalDevices());
	auto physicalDevice = pickPhysicalDevice(physicalDevices);

	auto familyIndex = getGraphicsQueueFamilyIndex(physicalDevice);
	returnValueOnSuccess(physicalDevice.getSurfaceSupportKHR(familyIndex, surface));


	auto format = getSwapchainFormat(physicalDevice, surface);

	auto device = createDevice(physicalDevice, familyIndex);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
	auto width = 0, height = 0;
	SDL_GetWindowSize(window, &width, &height);



	auto props = physicalDevice.getProperties();
	assert(props.limits.timestampComputeAndGraphics);


	auto surfaceCapabilities = returnValueOnSuccess(physicalDevice.getSurfaceCapabilitiesKHR(surface));

	
	auto acquireSemaphore = createSemaphore(device);
	auto releaseSemaphore = createSemaphore(device);
	auto fence = returnValueOnSuccess(device.createFence({}));

#if RTX
	auto meshMS = loadShader(device, "shaders/mesh.mesh.spv");

#else
	auto meshMS = loadShader(device, "shaders/triangle.vert.spv");

#endif


	auto meshFS = loadShader(device, "shaders/mesh.frag.spv");

	auto pipelineCache = nullptr;

	

	auto queue = device.getQueue(familyIndex, 0);

	auto renderPass = createRenderPass(device, format);
	auto swapchain = createSwapchain(device, surface, surfaceCapabilities, familyIndex, static_cast<U32>(width),
	                                 static_cast<U32>(height), format, renderPass, nullptr);


	auto queryPool = createQueryPool(device, 128);



	auto triangleLayout = createPipelineLayout(device);
	auto trianglePipeline = createGraphicsPipeline(device, pipelineCache, renderPass, triangleLayout,
	                                               meshMS, meshFS);
	
	auto commandPoolCreateInfo = vk::CommandPoolCreateInfo
	{
		.flags = vk::CommandPoolCreateFlagBits::eTransient,
		.queueFamilyIndex = familyIndex
	};
	auto commandPool = returnValueOnSuccess(device.createCommandPool(commandPoolCreateInfo));
	auto allocateInfo = vk::CommandBufferAllocateInfo
	{
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1
	};
	auto commandBuffer = vk::CommandBuffer{ returnValueOnSuccess(device.allocateCommandBuffers(allocateInfo)).front() };
	
	auto mesh = loadMesh("data/kitten.obj");


#if  RTX
	buildMeshlets(mesh);
#endif


	const auto memoryProperties = physicalDevice.getMemoryProperties();

	auto scratch = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);


	auto vb = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
	auto ib = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
#if  RTX
	auto mb = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
#endif


	uploadBuffer(device, commandPool, commandBuffer, queue, scratch, vb, mesh.vertices);
	uploadBuffer(device, commandPool, commandBuffer, queue, scratch, ib, mesh.indices);
#if  RTX
	uploadBuffer(device, commandPool, commandBuffer, queue, scratch, mb, mesh.meshlets);
#endif



	auto frameCpuAvg = 0.0;
	auto frameGpuAvg = 0.0;

	
	// Poll for user input.
	auto stillRunning = true;
	while (stillRunning)
	{
		auto frameCpuBegin = SDL_GetPerformanceCounter();

		auto event = SDL_Event{};
		while (SDL_PollEvent(&event))
		{
			switch (event.type)
			{

			case SDL_QUIT:
				stillRunning = false;
				break;
			case SDL_WINDOWEVENT:
				if(event.window.event == SDL_WINDOWEVENT_RESIZED && event.window.windowID == SDL_GetWindowID(window))
				{
					resizeSwapchain(device, surface, surfaceCapabilities, familyIndex,
					                static_cast<U32>(event.window.data1), static_cast<U32>(event.window.data2), format,
					                renderPass, swapchain);
					
				}
					
				break;

			default:
				// Do nothing.
				break;
			}
		}


		U32 imageIndex = 0;
		returnValueOnSuccess(device.acquireNextImageKHR(swapchain.swapchain, ~0ull, acquireSemaphore, nullptr, &imageIndex));

		returnValueOnSuccess(device.resetCommandPool(commandPool));

		auto beginInfo = vk::CommandBufferBeginInfo
		{
			.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
		};
		returnValueOnSuccess(commandBuffer.begin(beginInfo));

		commandBuffer.resetQueryPool(queryPool, 0,128);
		commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, 0);

		auto renderBeginBarrier = imageBarrier(swapchain.images[imageIndex], static_cast<vk::AccessFlagBits>(0),
			vk::AccessFlagBits::eColorAttachmentWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::DependencyFlagBits::eByRegion, nullptr, nullptr,
			std::array{ renderBeginBarrier });



		auto color = vk::ClearColorValue{ std::array{ 100.0f / 255.0f, 149.0f / 255.0f,237.0f / 255.0f,1.0f } };
		auto clearValue = vk::ClearValue{ color };
		auto renderPassBeginInfo = vk::RenderPassBeginInfo
		{
			.renderPass = renderPass,
			.framebuffer = swapchain.framebuffers[imageIndex],
			.renderArea = vk::Rect2D{.extent = vk::Extent2D{.width = swapchain.width, .height = swapchain.height}},
			.clearValueCount = 1,
			.pClearValues = &clearValue
		};

		commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		auto viewport = vk::Viewport
		{
			.x = 0,
			.y = static_cast<float>(swapchain.height),
			.width = static_cast<float>(swapchain.width),
			.height = -static_cast<float>(swapchain.height),
			.minDepth = 0,
			.maxDepth = 1
		};
		auto scissor = vk::Rect2D{ { 0, 0 }, { swapchain.width, swapchain.height } };
		commandBuffer.setViewport(0, 1, &viewport);
		commandBuffer.setScissor(0, 1, &scissor);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, trianglePipeline);


		const auto vbInfo = vk::DescriptorBufferInfo
		{
			.buffer = vb.buffer,
			.offset = 0,
			.range = vk::DeviceSize{ vb.size }
		};
#if RTX
		const auto mbInfo = vk::DescriptorBufferInfo
		{
			.buffer = mb.buffer,
			.offset = 0,
			.range = vk::DeviceSize{ mb.size }
		};


		const auto descriptors = std::array{
			vk::WriteDescriptorSet
			{
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &vbInfo
			},
			vk::WriteDescriptorSet
			{
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &mbInfo
			}
		};
		commandBuffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, triangleLayout, 0, descriptors);

		for(int i = 0; i < 100; i++)
		{
			commandBuffer.drawMeshTasksNV(static_cast<U32>(mesh.meshlets.size()), 0);
		}
		
		
#else

		const auto descriptors = std::array{
			vk::WriteDescriptorSet
			{
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &vbInfo
			}
		};
		commandBuffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, triangleLayout, 0, descriptors);

		auto dummyOffset = vk::DeviceSize{ 0 };
		commandBuffer.bindIndexBuffer(ib.buffer, dummyOffset, vk::IndexType::eUint32);
		//commandBuffer.draw(3, 1, 0, 0);

		for (int i = 0; i < 200; i++)
		{
			commandBuffer.drawIndexed(static_cast<U32>(mesh.indices.size()), 1, 0, 0, 0);

		}
#endif

		commandBuffer.endRenderPass();


		auto renderEndBarrier = imageBarrier(swapchain.images[imageIndex],
			vk::AccessFlagBits::eColorAttachmentWrite,
			static_cast<vk::AccessFlagBits>(0), vk::ImageLayout::eUndefined,
			vk::ImageLayout::ePresentSrcKHR);

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlagBits::eByRegion,
			nullptr, nullptr, std::array{ renderEndBarrier });

		commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, 1);

		returnValueOnSuccess(commandBuffer.end());


		auto stageMask = vk::PipelineStageFlags{ vk::PipelineStageFlagBits::eColorAttachmentOutput };

		auto submits = std::array<vk::SubmitInfo, 1>
		{
			vk::SubmitInfo
			{
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &acquireSemaphore,
				.pWaitDstStageMask = &stageMask,
				.commandBufferCount = 1,
				.pCommandBuffers = &commandBuffer,
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &releaseSemaphore
			}
		};
		returnValueOnSuccess(queue.submit(submits));

		auto presentInfo = vk::PresentInfoKHR
		{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &releaseSemaphore,
			.swapchainCount = 1,
			.pSwapchains = &swapchain.swapchain,
			.pImageIndices = &imageIndex
		};

		returnValueOnSuccess(queue.presentKHR(presentInfo));
		returnValueOnSuccess(device.waitIdle());

		auto queryResults = std::array<uint64_t, 2>{};
		returnValueOnSuccess(device.getQueryPoolResults(queryPool, 0, 2, sizeof(queryResults), queryResults.data(), sizeof(queryResults[0]), vk::QueryResultFlagBits::e64));

		auto frameGpuBegin = static_cast<double>(queryResults[0]);
		auto frameGpuEnd = static_cast<double>(queryResults[1]);


		auto frameCpuEnd = SDL_GetPerformanceCounter();
		auto freq = SDL_GetPerformanceFrequency();

		auto cpuTime = static_cast<double>(frameCpuEnd - frameCpuBegin) / static_cast<double>(freq) * 1000.0;
		auto gpuTime = (frameGpuEnd - frameGpuBegin) * static_cast<double>(props.limits.timestampPeriod) * 1e-6;


		frameCpuAvg = frameCpuAvg * 0.95 + cpuTime * 0.05;
		frameGpuAvg = frameGpuAvg * 0.95 + gpuTime * 0.05;

		auto title = std::stringstream{};
		title << "cpu: " << std::fixed << std::setprecision(2) << frameCpuAvg << " ms" << "    gpu: " << frameGpuAvg << " ms";
	
		SDL_SetWindowTitle(window, title.str().c_str());


		//SDL_WaitEvent(nullptr);
	}


	returnValueOnSuccess(device.waitIdle());
	// Clean up.

	destroyBuffer(scratch, device);
	destroyBuffer(vb, device);
	destroyBuffer(ib, device);
#if RTX
	destroyBuffer(mb, device);
#endif

	device.destroySwapchainKHR(swapchain.swapchain);
	device.destroyPipeline(trianglePipeline);
	device.destroySemaphore(acquireSemaphore);
	device.destroySemaphore(releaseSemaphore);
	device.destroyFence(fence);
	device.destroyCommandPool(commandPool);
	device.destroyQueryPool(queryPool);
	//device.destroyDescriptorSetLayout()
	device.destroyPipelineLayout(triangleLayout);
	for(auto &framebuffer : swapchain.framebuffers)
	{
		device.destroy(framebuffer);
	}
	for(auto &imageView : swapchain.imageViews)
	{
		device.destroyImageView(imageView);
	}
	device.destroyShaderModule(meshFS);
	device.destroyShaderModule(meshMS);
	device.destroyRenderPass(renderPass);
	device.destroy();

	
	instance.destroySurfaceKHR(surface);
	SDL_DestroyWindow(window);
	SDL_Quit();

	instance.destroyDebugReportCallbackEXT(debugCallback);
	instance.destroy();

	return 0;
}
