
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
#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.hpp>
#pragma warning(pop)


#include <iostream>
#include <vector>

using U32 = uint32_t;

template<typename T>
T returnValueOnSuccess(const vk::ResultValue<T>& call) { assert(call.result == vk::Result::eSuccess); return call.value; }


vk::Result returnValueOnSuccess(const vk::Result& result) { assert(result == vk::Result::eSuccess); return result; }

vk::PhysicalDevice pickPhysicalDevice(const std::vector<vk::PhysicalDevice>& physicalDevices)
{
	assert(!physicalDevices.empty());
	for (auto physicalDevice : physicalDevices)
	{
		auto props = physicalDevice.getProperties();

		if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		{
			std::cout << "Picking discrete GPU: " << props.deviceName << "\n";
			return physicalDevice;
		}
	}

	auto props = physicalDevices[0].getProperties();
	std::cout << "Picking fallback GPU: " << props.deviceName << "\n";
	return physicalDevices[0];
}

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

U32 getGraphicsQueueFamilyIndex(vk::PhysicalDevice physicalDevice)
{
	const auto properties = physicalDevice.getQueueFamilyProperties();
	for (U32 i = 0; i < properties.size(); i++)
	{
		if (properties[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			return i;
		}
	}
	BOOST_ASSERT_MSG(true, L"No graphics queue found!");
	return 0;
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


	auto extensions = std::array<const char*, 1>{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	const auto deviceCreateInfo = vk::DeviceCreateInfo
	{
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<U32>(extensions.size()),
		.ppEnabledExtensionNames = extensions.data()
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

vk::SwapchainKHR crateSwapchain(const vk::Device device, vk::SurfaceKHR surface, vk::SurfaceCapabilitiesKHR surfaceCapabilities, U32 familyIndex, U32 width, U32 height, vk::Format format)
{
	
	auto supportedCompositeAlpha =
		surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque ? vk::CompositeAlphaFlagBitsKHR::eOpaque :
		surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied ? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied :
		surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied ? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied :
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
		.presentMode = vk::PresentModeKHR::eFifo
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

vk::Framebuffer createFramebuffer(const vk::Device device, vk::RenderPass renderPass, vk::ImageView imageView, U32 width, U32 height)
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
	const auto createInfo = vk::PipelineLayoutCreateInfo
	{

	};

	return returnValueOnSuccess(device.createPipelineLayout(createInfo));
}

vk::Pipeline createGraphicsPipeline(vk::Device device, vk::PipelineCache pipelineCache, vk::RenderPass renderPass, vk::PipelineLayout pipelineLayout, vk::ShaderModule triangleVertexShader, vk::ShaderModule triangleFragmentShader)
{
	auto stages = std::array<vk::PipelineShaderStageCreateInfo, 2>
	{
		vk::PipelineShaderStageCreateInfo
		{
			.stage = vk::ShaderStageFlagBits::eVertex,
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

	auto vertexInput = vk::PipelineVertexInputStateCreateInfo
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
		.pVertexInputState = &vertexInput,
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


VkBool32 debugReportCallback(
	VkDebugReportFlagsEXT flags,
	[[maybe_unused]] VkDebugReportObjectTypeEXT objectType,
	[[maybe_unused]] uint64_t object,
	[[maybe_unused]] size_t location,
	[[maybe_unused]] int32_t messageCode,
	[[maybe_unused]] const char* pLayerPrefix,
	const char* pMessage,
	[[maybe_unused]] void* pUserData)
{
	
	const auto level = vk::to_string(flags & static_cast<U32>(vk::DebugReportFlagBitsEXT::eError) ? vk::DebugReportFlagBitsEXT::eError
		                                 : flags & (static_cast<U32>(vk::DebugReportFlagBitsEXT::eWarning) | static_cast<U32>(vk::DebugReportFlagBitsEXT::ePerformanceWarning)) ? vk::DebugReportFlagBitsEXT::eWarning
		                                 : vk::DebugReportFlagBitsEXT::eInformation);

	std::cout << "[" << level << "]: " << pMessage << std::endl;

	//BOOST_ASSERT_MSG(flags & static_cast<U32>(vk::DebugReportFlagBitsEXT::eError), L"Validation error encountered!");
	return VK_FALSE;
}

vk::DebugReportCallbackEXT registerDebugCallback(const vk::Instance instance)
{
	const auto createInfo = vk::DebugReportCallbackCreateInfoEXT
	{
		.flags = vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning | vk::DebugReportFlagBitsEXT::ePerformanceWarning | vk::DebugReportFlagBitsEXT::eInformation,
		.pfnCallback = debugReportCallback
	};


	//instance.getProcAddr("vkCreateDebugReportCallbackEXT");
	return returnValueOnSuccess(instance.createDebugReportCallbackEXT(createInfo));
}

vk::ImageMemoryBarrier imageBarrier(vk::Image image,
                                    [[maybe_unused]] vk::PipelineStageFlags srcStageMask,
	vk::AccessFlags srcAccessMask,
	vk::ImageLayout oldLayout,
	[[maybe_unused]] vk::PipelineStageFlags dstStageMask,
	vk::AccessFlags dstAccessMask,
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

void pipelineBarrierImage([[maybe_unused]] vk::CommandBuffer commandBuffer, [[maybe_unused]] vk::ImageMemoryBarrier barrier)
{
	//commandBuffer.pipelineBarrier()
}

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

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
		SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_VULKAN);
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


	auto surfaceCapabilities = returnValueOnSuccess(physicalDevice.getSurfaceCapabilitiesKHR(surface));

	
	auto swapchain = crateSwapchain(device, surface, surfaceCapabilities, familyIndex, static_cast<U32>(width), static_cast<U32>(height), format);
	auto acquireSemaphore = createSemaphore(device);
	auto releaseSemaphore = createSemaphore(device);

	auto fence = returnValueOnSuccess(device.createFence({}));




	auto triangleVertexShader = loadShader(device, "shaders/triangle.vert.spv");
	auto triangleFragmentShader = loadShader(device, "shaders/triangle.frag.spv");

	auto pipelineCache = nullptr;

	

	auto queue = device.getQueue(familyIndex, 0);

	auto swapchainImage = returnValueOnSuccess(device.getSwapchainImagesKHR(swapchain));

	auto swapchainImageViews = std::vector<vk::ImageView>{ swapchainImage.size() };
	auto swapchainFramebuffer = std::vector<vk::Framebuffer>{ swapchainImage.size() };
	auto renderPass = createRenderPass(device, format);

	auto pipelineLayout = createPipelineLayout(device);
	auto trianglePipeline = createGraphicsPipeline(device, pipelineCache, renderPass, pipelineLayout, triangleVertexShader, triangleFragmentShader);

	for (U32 i = 0; i < swapchainImageViews.size(); i++)
	{
		swapchainImageViews[i] = createImageView(device, swapchainImage[i], format);
	}
	
	for (U32 i = 0; i < swapchainFramebuffer.size(); i++)
	{
		swapchainFramebuffer[i] = createFramebuffer(device, renderPass, swapchainImageViews[i], static_cast<U32>(width), static_cast<U32>(height));
	}


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

	// Poll for user input.
	auto stillRunning = true;
	while (stillRunning)
	{

		auto event = SDL_Event{};
		while (SDL_PollEvent(&event))
		{
			U32 imageIndex = 0;
			returnValueOnSuccess(device.acquireNextImageKHR(swapchain, ~0ull, acquireSemaphore, nullptr, &imageIndex));

			returnValueOnSuccess(device.resetCommandPool(commandPool));

			auto beginInfo = vk::CommandBufferBeginInfo
			{
				.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
			};
			returnValueOnSuccess(commandBuffer.begin(beginInfo));
			
			auto color = vk::ClearColorValue{ std::array{ 100.0f/255.0f, 149.0f/255.0f,237.0f/255.0f,1.0f } };
			auto clearValue = vk::ClearValue{ color };
			auto renderPassBeginInfo = vk::RenderPassBeginInfo
			{
				.renderPass = renderPass,
				.framebuffer = swapchainFramebuffer[imageIndex],
				.renderArea = vk::Rect2D{.extent = vk::Extent2D{.width = static_cast<U32>(width), .height = static_cast<U32>(height)}},
				.clearValueCount = 1,
				.pClearValues = &clearValue
			};

			commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			auto viewport = vk::Viewport
			{
				.x = 0,
				.y = static_cast<float>(height),
				.width = static_cast<float>(width),
				.height = -static_cast<float>(height),
				.minDepth = 0,
				.maxDepth = 1
			};
			auto scissor = vk::Rect2D{ { 0, 0 }, { static_cast<U32>(width), static_cast<U32>(height) } };
			commandBuffer.setViewport(0, 1, &viewport);
			commandBuffer.setScissor(0, 1, &scissor);
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, trianglePipeline);
			commandBuffer.draw(3, 1, 0, 0);
			
			commandBuffer.endRenderPass();

			

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
				.pSwapchains = &swapchain,
				.pImageIndices = &imageIndex
			};
			
			returnValueOnSuccess(queue.presentKHR(presentInfo));
			returnValueOnSuccess(device.waitIdle());

			switch (event.type)
			{

			case SDL_QUIT:
				stillRunning = false;
				break;

			default:
				// Do nothing.
				break;
			}
		}
		SDL_WaitEvent(nullptr);
		//SDL_Delay(10);
	}

	// Clean up.
	device.destroySwapchainKHR(swapchain);
	device.destroyPipeline(trianglePipeline);
	device.destroySemaphore(acquireSemaphore);
	device.destroySemaphore(releaseSemaphore);
	device.destroyFence(fence);
	device.destroyCommandPool(commandPool);
	device.destroyPipelineLayout(pipelineLayout);
	for(auto &framebuffer : swapchainFramebuffer)
	{
		device.destroy(framebuffer);
	}
	for(auto &imageView : swapchainImageViews)
	{
		device.destroyImageView(imageView);
	}
	device.destroyShaderModule(triangleFragmentShader);
	device.destroyShaderModule(triangleVertexShader);
	device.destroyRenderPass(renderPass);
	device.destroy();

	
	instance.destroySurfaceKHR(surface);
	SDL_DestroyWindow(window);
	SDL_Quit();

	instance.destroyDebugReportCallbackEXT(debugCallback);
	instance.destroy();

	return 0;
}
