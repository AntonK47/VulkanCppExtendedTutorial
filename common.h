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

template<typename T>
T returnValueOnSuccess(const vk::ResultValue<T>& call) { assert(call.result == vk::Result::eSuccess); return call.value; }


inline void returnValueOnSuccess(const vk::Result& result) { assert(result == vk::Result::eSuccess); }

using U32 = uint32_t;