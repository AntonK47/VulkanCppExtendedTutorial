#include "common.h"
#include "swapchain.h"

namespace 
{
	vk::SwapchainKHR createSwapchainImpl(const vk::Device device, vk::SurfaceKHR surface,
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
}



void tut::swapchain::destroySwapchain(const vk::Device device, const Swapchain& swapchain)
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

tut::swapchain::Swapchain tut::swapchain::createSwapchain(const vk::Device device, const vk::SurfaceKHR surface,
                                                          const vk::SurfaceCapabilitiesKHR surfaceCapabilities, const U32 familyIndex, const U32 width,
                                                          const U32 height, const vk::Format format, const vk::RenderPass renderPass,
                                                          const vk::SwapchainKHR oldSwapchain)
{
	auto swapchain = createSwapchainImpl(device, surface, surfaceCapabilities, familyIndex, width,
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

void tut::swapchain::resizeSwapchain(const vk::Device device, const vk::SurfaceKHR surface,
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
