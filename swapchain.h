#pragma once


namespace tut::swapchain
{
	struct Swapchain
	{
		vk::SwapchainKHR swapchain;
		std::vector<vk::Image> images;
		std::vector<vk::Framebuffer> framebuffers;
		std::vector<vk::ImageView> imageViews;

		U32 width;
		U32 height;
	};

	vk::SwapchainKHR createSwapchain(const vk::Device device, vk::SurfaceKHR surface,
		vk::SurfaceCapabilitiesKHR surfaceCapabilities, U32 familyIndex, U32 width, U32 height,
		vk::Format format, vk::SwapchainKHR oldSwapchain);
	void destroySwapchain(const vk::Device device, const Swapchain& swapchain);

	Swapchain createSwapchain(const vk::Device device, const vk::SurfaceKHR surface,
		const vk::SurfaceCapabilitiesKHR surfaceCapabilities, const U32 familyIndex, const U32 width,
		const U32 height, const vk::Format format, const vk::RenderPass renderPass,
		const vk::SwapchainKHR oldSwapchain);
	void resizeSwapchain(const vk::Device device, const vk::SurfaceKHR surface,
		const vk::SurfaceCapabilitiesKHR surfaceCapabilities, const U32 familyIndex, const U32 width,
		const U32 height, const vk::Format format, const vk::RenderPass renderPass,
		Swapchain& swapchain);
}