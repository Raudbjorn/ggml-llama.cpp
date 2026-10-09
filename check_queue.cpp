#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>

int main() {
    zeInit(0);
    ze_driver_handle_t driver;
    uint32_t count = 1;
    zeDriverGet(&count, &driver);
    ze_device_handle_t device;
    zeDeviceGet(driver, &count, &device);
    
    uint32_t group_count = 0;
    zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr);
    std::vector<ze_command_queue_group_properties_t> props(group_count);
    zeDeviceGetCommandQueueGroupProperties(device, &group_count, props.data());
    
    for(uint32_t i=0; i<group_count; ++i) {
        printf("Group %d: flags=%d\n", i, props[i].flags);
        if(props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
            printf("Group %d is Copy Engine\n", i);
        }
    }
    return 0;
}
