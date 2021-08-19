#pragma once

#pragma warning(push)
#pragma warning( disable : 26819 )
#pragma warning( disable : 26812 )
#pragma warning( disable : 28251 )
#pragma warning( disable : 26812 )
#pragma warning( disable : 26495 )
#pragma warning( disable : 4464 )
#pragma warning( disable : 4820 )
#include <boost/assert.hpp>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS

#include <vulkan/vulkan.hpp>
#pragma warning(pop)

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE


template<typename T>
T returnValueOnSuccess(const vk::ResultValue<T>& call) { assert(call.result == vk::Result::eSuccess); return call.value; }


inline void returnValueOnSuccess(const vk::Result& result) { assert(result == vk::Result::eSuccess); }

using U32 = uint32_t;


inline vk::Framebuffer createFramebuffer(const vk::Device device, vk::RenderPass renderPass, vk::ImageView imageView,
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

inline vk::ImageView createImageView(const vk::Device device, vk::Image image, vk::Format format)
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