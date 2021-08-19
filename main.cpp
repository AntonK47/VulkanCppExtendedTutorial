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


#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1  // NOLINT(cppcoreguidelines-macro-usage)

#pragma warning(push)
#pragma warning( disable : 26819 )
#pragma warning( disable : 26451 )
#pragma warning( disable : 26812 )
#pragma warning( disable : 28251 )
#pragma warning( disable : 26812 )
#pragma warning( disable : 26495 )
#pragma warning( disable : 4464 )
#pragma warning( disable : 4820 )

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define FAST_OBJ_IMPLEMENTATION
#include <fast_obj.h>
#include <meshoptimizer.h>
#pragma warning(pop)

#include <iostream>
#include <iomanip>
#include <vector>

#include "common.h"
#include "spirv_reflect.h"
#include "shaders.h"
#include "swapchain.h"



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

vk::Device createDevice(const vk::PhysicalDevice physicalDevice, U32 familyIndex, const bool rtxSupported)
{
	auto queuePriorities = std::array<float, 1>{ 1.0f };
	auto deviceQueueCreateInfo = vk::DeviceQueueCreateInfo
	{
		.queueFamilyIndex = familyIndex,
		.queueCount = 1,
		.pQueuePriorities = queuePriorities.data()
	};


	auto extensions = std::vector
	{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
		VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
		VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME
	};
	if(rtxSupported)
	{
		extensions.push_back(VK_NV_MESH_SHADER_EXTENSION_NAME);
	}
	auto deviceCreateInfo = vk::DeviceCreateInfo{};

	if(rtxSupported)
	{
		const auto features = vk::StructureChain<vk::PhysicalDeviceFeatures2,
			vk::PhysicalDevice16BitStorageFeatures,
			vk::PhysicalDevice8BitStorageFeatures,
			vk::PhysicalDeviceShaderFloat16Int8Features,
			vk::PhysicalDeviceMeshShaderFeaturesNV,
			vk::PhysicalDeviceSynchronization2FeaturesKHR>
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
			vk::PhysicalDeviceMeshShaderFeaturesNV
			{
				.taskShader = vk::Bool32{rtxSupported},
				.meshShader = vk::Bool32{rtxSupported}
			},
			vk::PhysicalDeviceSynchronization2FeaturesKHR
			{
				.synchronization2 = vk::Bool32{true}
			}
		};

		deviceCreateInfo = vk::DeviceCreateInfo
		{
			.pNext = &features.get(),
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &deviceQueueCreateInfo,
			.enabledExtensionCount = static_cast<U32>(extensions.size()),
			.ppEnabledExtensionNames = extensions.data(),

		};
	}
	else
	{
		const auto features = vk::StructureChain<vk::PhysicalDeviceFeatures2,
			vk::PhysicalDevice16BitStorageFeatures,
			vk::PhysicalDevice8BitStorageFeatures,
			vk::PhysicalDeviceShaderFloat16Int8Features,
			vk::PhysicalDeviceSynchronization2FeaturesKHR>
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
			vk::PhysicalDeviceSynchronization2FeaturesKHR
			{
				.synchronization2 = vk::Bool32{true}
			}
		};

		deviceCreateInfo = vk::DeviceCreateInfo
		{
			.pNext = &features.get(),
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &deviceQueueCreateInfo,
			.enabledExtensionCount = static_cast<U32>(extensions.size()),
			.ppEnabledExtensionNames = extensions.data(),

		};
	}

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





struct Vertex
{
	uint16_t vx, vy, vz, vw;
	uint8_t nx, ny, nz;
	uint16_t tu, tv;
};

struct alignas(16) Meshlet
{
	float cone[4];
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

float halfToFloat(uint16_t v)
{
	const uint16_t sign = v >> 15;
	const uint16_t exp = (v >> 10) & 31;
	const uint16_t mant = v & 1023;

	assert(exp != 31);

	if(exp == 0)
	{
		assert(mant == 0);
		return 0.0f;
	}
	else
	{
		return (sign ? -1.0f : 1.0f) * ldexpf(static_cast<float>(mant + 1024) / 1024.0f, exp - 15);
	}
}

void buildMeshletsCones(Mesh& mesh)
{
	for(auto& meshlet : mesh.meshlets)
	{
		float normals[126][3] = {};

		for(U32 i = 0; i < meshlet.triangleCount; i++)
		{
			const U32 a = meshlet.indices[i * 3 + 0];
			const U32 b = meshlet.indices[i * 3 + 1];
			const U32 c = meshlet.indices[i * 3 + 2];

			const auto& va = mesh.vertices[meshlet.vertices[a]];
			const auto& vb = mesh.vertices[meshlet.vertices[b]];
			const auto& vc = mesh.vertices[meshlet.vertices[c]];

			const float p0[3] = { halfToFloat(va.vx), halfToFloat(va.vy), halfToFloat(va.vz) };
			const float p1[3] = { halfToFloat(vb.vx), halfToFloat(vb.vy), halfToFloat(vb.vz) };
			const float p2[3] = { halfToFloat(vc.vx), halfToFloat(vc.vy), halfToFloat(vc.vz) };

			const float p10[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
			const float p20[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };

			const float normalX = p10[1] * p20[2] - p10[2] * p20[1];
			const float normalY = p10[2] * p20[0] - p10[0] * p20[2];
			const float normalZ = p10[0] * p20[1] - p10[1] * p20[0];

			const float length = sqrtf(normalX * normalX + normalY * normalY + normalZ * normalZ);

			const float invLength = length == 0.0f ? 0.0f : 1 / length;

			normals[i][0] = normalX * invLength;
			normals[i][1] = normalY * invLength;
			normals[i][2] = normalZ * invLength;
		}

		float avgNormal[3] = {};

		for (U32 i = 0; i < meshlet.triangleCount; i++)
		{
			avgNormal[0] += normals[i][0];
			avgNormal[1] += normals[i][1];
			avgNormal[2] += normals[i][2];
		}

		const float avgLength = sqrtf(avgNormal[0] * avgNormal[0] + avgNormal[1] * avgNormal[1] + avgNormal[2] * avgNormal[2]);


		if(avgLength == 0.0f)
		{
			avgNormal[0] = 1.0f;
			avgNormal[1] = 0.0f;
			avgNormal[2] = 0.0f;
		}
		else
		{
			avgNormal[0] /= avgLength;
			avgNormal[1] /= avgLength;
			avgNormal[2] /= avgLength;
		}

		// ReSharper disable once IdentifierTypo
		float mindp = 1.0f;
		for (U32 i = 0; i < meshlet.triangleCount; i++)
		{
			float dp = normals[i][0] * avgNormal[0] + normals[i][1] * avgNormal[1] + normals[i][2] * avgNormal[2];

			mindp = std::min(mindp, dp);
		}

		meshlet.cone[0] = avgNormal[0];
		meshlet.cone[1] = avgNormal[1];
		meshlet.cone[2] = avgNormal[2];
		if (mindp <= 0.0f)
			meshlet.cone[3] = 1.0f;
		else
			meshlet.cone[3] = sqrtf(1.0f - mindp * mindp);

	}
}

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
				0,
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

	void* data = 0;
	if(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)
	{
		data = returnValueOnSuccess(device.mapMemory(memory, 0, size));
	}

	return Buffer
	{
		.buffer = buffer,
		.memory = memory,
		.data = data,
		.size = size
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

vk::ImageMemoryBarrier2KHR createImageSynchronizationBarrier(std::vector<vk::Image>::const_reference image, vk::PipelineStageFlagBits2KHR srcStageMask, vk::AccessFlagBits2KHR srcAccessMask,
	vk::PipelineStageFlagBits2KHR dstStageMask, vk::AccessFlagBits2KHR dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
	return vk::ImageMemoryBarrier2KHR
	{
		.srcStageMask = srcStageMask,
		.srcAccessMask = srcAccessMask,
		.dstStageMask = dstStageMask,
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
	//provides VK_KHR_synchronization2 extension even if the underlying driver do not provide the extension
	layers.push_back("VK_LAYER_KHRONOS_synchronization2");
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

	auto deviceExtensions = returnValueOnSuccess(physicalDevice.enumerateDeviceExtensionProperties());

	auto rtxSupported = false;

	for(auto& ext: deviceExtensions)
	{
		if(strcmp(ext.extensionName.data(), "VK_NV_mesh_shader") == 0)
		{
			rtxSupported = true;
			break;
		}
	}
	bool rtxEnabled = rtxSupported;
	
	auto familyIndex = getGraphicsQueueFamilyIndex(physicalDevice);
	returnValueOnSuccess(physicalDevice.getSurfaceSupportKHR(familyIndex, surface));


	auto format = getSwapchainFormat(physicalDevice, surface);

	auto device = createDevice(physicalDevice, familyIndex, rtxSupported);
	
	auto width = 0, height = 0;
	SDL_GetWindowSize(window, &width, &height);
	

	auto props = physicalDevice.getProperties();
	assert(props.limits.timestampComputeAndGraphics);


	auto surfaceCapabilities = returnValueOnSuccess(physicalDevice.getSurfaceCapabilitiesKHR(surface));

	
	auto acquireSemaphore = createSemaphore(device);
	auto releaseSemaphore = createSemaphore(device);
	auto fence = returnValueOnSuccess(device.createFence({}));
	
	tut::shaders::Shader meshMs;
	if (rtxSupported)
	{
		meshMs = tut::shaders::loadShader(device, "shaders/mesh.mesh.spv");
	}


	auto meshVs = tut::shaders::loadShader(device, "shaders/triangle.vert.spv");
	auto meshFs = tut::shaders::loadShader(device, "shaders/mesh.frag.spv");



	auto pipelineCache = nullptr;
	auto queue = device.getQueue(familyIndex, 0);

	auto renderPass = createRenderPass(device, format);
	auto swapchain = tut::swapchain::createSwapchain(device, surface, surfaceCapabilities, familyIndex, static_cast<U32>(width),
	                                                 static_cast<U32>(height), format, renderPass, nullptr);


	auto queryPool = createQueryPool(device, 128);


	auto setLayout = tut::shaders::createSetLayout(device, { meshFs, meshVs });
	auto meshLayout = tut::shaders::createPipelineLayout(device, setLayout);
	auto descriptorTemplate = tut::shaders::createUpdateTemplate(device, vk::PipelineBindPoint::eGraphics, setLayout, meshLayout, { meshFs, meshVs });
	auto meshPipeline = tut::shaders::createGraphicsPipeline(device, pipelineCache, renderPass, meshLayout,{meshVs,meshFs});
	auto meshPipelineRtx = vk::Pipeline{};
	auto meshLayoutRtx = vk::PipelineLayout{};
	auto setLayoutRtx = vk::DescriptorSetLayout{};
	auto descriptorTemplateRtx = vk::DescriptorUpdateTemplate{};
	if(rtxSupported)
	{
		setLayoutRtx = tut::shaders::createSetLayout(device, { meshFs, meshMs });
		meshLayoutRtx = tut::shaders::createPipelineLayout(device, setLayoutRtx);
		meshPipelineRtx = tut::shaders::createGraphicsPipeline(device, pipelineCache, renderPass, meshLayoutRtx, { meshMs,meshFs });
		descriptorTemplateRtx = tut::shaders::createUpdateTemplate(device, vk::PipelineBindPoint::eGraphics, setLayoutRtx, meshLayoutRtx, { meshFs, meshMs });
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
	
	auto mesh = loadMesh("data/kitten.obj");


	if (rtxSupported)
	{
		buildMeshlets(mesh);
		buildMeshletsCones(mesh);
	}


	const auto memoryProperties = physicalDevice.getMemoryProperties();

	auto scratch = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);


	auto vb = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
	auto ib = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);



	uploadBuffer(device, commandPool, commandBuffer, queue, scratch, vb, mesh.vertices);
	uploadBuffer(device, commandPool, commandBuffer, queue, scratch, ib, mesh.indices);

	auto mb = Buffer{};
	if(rtxSupported)
	{
		mb = createBuffer(device, memoryProperties, 1024 * 1024 * 1024, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
		uploadBuffer(device, commandPool, commandBuffer, queue, scratch, mb, mesh.meshlets);
	}




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
			case SDL_KEYDOWN:
				switch (event.key.keysym.sym)
				{
				case SDLK_SPACE:
					rtxEnabled = !rtxEnabled;
					break;
				default:
					break;
				}
				break;

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
		commandBuffer.writeTimestamp2KHR(vk::PipelineStageFlagBits2KHR::eBottomOfPipe, queryPool, 0);


		const auto renderBeginBarrier = createImageSynchronizationBarrier(swapchain.images[imageIndex],
			vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput, vk::AccessFlagBits2KHR::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput, vk::AccessFlagBits2KHR::eColorAttachmentWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

		auto const renderBeginSyncDependency = vk::DependencyInfoKHR
		{
			.dependencyFlags = vk::DependencyFlagBits::eByRegion,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &renderBeginBarrier
		};
		commandBuffer.pipelineBarrier2KHR(renderBeginSyncDependency);



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
	
		
		
		if (rtxEnabled)
		{
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, meshPipelineRtx);
			
			const auto descriptors = std::array
			{
				tut::shaders::DescriptorInfo
				{
					.buffer =  vk::DescriptorBufferInfo
					{
						vb.buffer,
						0,
						vb.size
					}
				},
				tut::shaders::DescriptorInfo
				{
					.buffer = vk::DescriptorBufferInfo
					{
						mb.buffer,
						0,
						mb.size
					}
				},
			};
			

			commandBuffer.pushDescriptorSetWithTemplateKHR(descriptorTemplateRtx, meshLayoutRtx, 0, &descriptors);

			for(int i = 0; i < 4000; i++)
			{
				commandBuffer.drawMeshTasksNV(static_cast<U32>(mesh.meshlets.size()), 0);
			}
		}
		else
		{
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, meshPipeline);

			const auto descriptors = std::array
			{
				tut::shaders::DescriptorInfo
				{
					.buffer = vk::DescriptorBufferInfo
					{
						vb.buffer,
						0,
						vb.size
					}
				}
			};


			commandBuffer.pushDescriptorSetWithTemplateKHR(descriptorTemplate, meshLayout, 0, &descriptors);

			auto dummyOffset = vk::DeviceSize{ 0 };
			commandBuffer.bindIndexBuffer(ib.buffer, dummyOffset, vk::IndexType::eUint32);

			for (int i = 0; i < 4000; i++)
			{
				commandBuffer.drawIndexed(static_cast<U32>(mesh.indices.size()), 1, 0, 0, 0);

			}
		}

		commandBuffer.endRenderPass();
		

		const auto renderEndBarrier = createImageSynchronizationBarrier(swapchain.images[imageIndex], vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput, vk::AccessFlagBits2KHR::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2KHR::eAllCommands, vk::AccessFlagBits2KHR::eNone, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);


		auto const renderEndSyncDependency = vk::DependencyInfoKHR
		{
			.dependencyFlags = vk::DependencyFlagBits::eByRegion,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &renderEndBarrier
		};
		
		commandBuffer.pipelineBarrier2KHR(renderEndSyncDependency);
		
		commandBuffer.writeTimestamp2KHR(vk::PipelineStageFlagBits2KHR::eBottomOfPipe, queryPool, 1);

		returnValueOnSuccess(commandBuffer.end());


		const auto acquireCompleteInfo = vk::SemaphoreSubmitInfoKHR
		{
			.semaphore = acquireSemaphore,
			.stageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput
		};

		const auto renderingCompleteInfo = vk::SemaphoreSubmitInfoKHR
		{
			.semaphore = releaseSemaphore,
			.stageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput
		};

		const auto commandBufferSubmitInfo = vk::CommandBufferSubmitInfoKHR
		{
			.commandBuffer = commandBuffer
		};

		const auto submits = std::array
		{
			vk::SubmitInfo2KHR
			{
				.waitSemaphoreInfoCount = 1,
				.pWaitSemaphoreInfos = &acquireCompleteInfo,
				.commandBufferInfoCount = 1,
				.pCommandBufferInfos = &commandBufferSubmitInfo,
				.signalSemaphoreInfoCount = 1,
				.pSignalSemaphoreInfos = &renderingCompleteInfo

			}
		};
		returnValueOnSuccess(queue.submit2KHR(submits));

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
		title << "cpu: " << std::fixed << std::setprecision(2) << frameCpuAvg << " ms" << "    gpu: " << frameGpuAvg << " ms" << " RTX " << (rtxEnabled ? "ON":"OFF");
	
		SDL_SetWindowTitle(window, title.str().c_str());


		//SDL_WaitEvent(nullptr);
	}


	returnValueOnSuccess(device.waitIdle());
	// Clean up.

	destroyBuffer(scratch, device);
	destroyBuffer(vb, device);
	destroyBuffer(ib, device);
	if (rtxSupported)
	{
		destroyBuffer(mb, device);
	}

	device.destroySwapchainKHR(swapchain.swapchain);
	device.destroyPipeline(meshPipeline);
	if(rtxSupported)
	{
		device.destroyPipeline(meshPipelineRtx);
	}
	device.destroySemaphore(acquireSemaphore);
	device.destroySemaphore(releaseSemaphore);
	device.destroyFence(fence);
	device.destroyCommandPool(commandPool);
	device.destroyQueryPool(queryPool);
	device.destroyDescriptorSetLayout(setLayout);
	device.destroyDescriptorUpdateTemplate(descriptorTemplate);
	if(rtxSupported)
	{
		device.destroyDescriptorSetLayout(setLayoutRtx);
		device.destroyDescriptorUpdateTemplate(descriptorTemplateRtx);
	}
	device.destroyPipelineLayout(meshLayout);
	if(rtxSupported)
	{
		device.destroyPipelineLayout(meshLayoutRtx);
	}
	for(auto &framebuffer : swapchain.framebuffers)
	{
		device.destroy(framebuffer);
	}
	for(auto &imageView : swapchain.imageViews)
	{
		device.destroyImageView(imageView);
	}
	device.destroyShaderModule(meshFs.module);
	device.destroyShaderModule(meshVs.module);
	if(rtxSupported)
	{
		device.destroyShaderModule(meshMs.module);
	}
	device.destroyRenderPass(renderPass);
	device.destroy();

	
	instance.destroySurfaceKHR(surface);
	SDL_DestroyWindow(window);
	SDL_Quit();

	instance.destroyDebugReportCallbackEXT(debugCallback);
	instance.destroy();

	return 0;
}
