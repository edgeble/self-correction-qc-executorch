// print what the QNN HTP runtime reports for this device: SoC model id/name, Hexagon arch
#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <dlfcn.h>
#include <stdio.h>
int main() {
    void *lib = dlopen("libQnnHtp.so", RTLD_NOW);
    if (!lib) { printf("dlopen libQnnHtp.so: %s\n", dlerror()); return 1; }
    typedef Qnn_ErrorHandle_t (*GetProv)(const QnnInterface_t ***, uint32_t *);
    GetProv gp = (GetProv)dlsym(lib, "QnnInterface_getProviders");
    const QnnInterface_t **prov = nullptr; uint32_t n = 0;
    if (!gp || gp(&prov, &n) != QNN_SUCCESS || n == 0) { printf("no providers\n"); return 1; }
    QNN_INTERFACE_VER_TYPE q = prov[0]->QNN_INTERFACE_VER_NAME;
    printf("QNN API %u.%u.%u, backend %s\n", prov[0]->apiVersion.coreApiVersion.major, prov[0]->apiVersion.coreApiVersion.minor,
           prov[0]->apiVersion.coreApiVersion.patch, prov[0]->providerName);
    Qnn_BackendHandle_t be = nullptr;
    Qnn_ErrorHandle_t e = q.backendCreate(nullptr, nullptr, &be);
    printf("backendCreate: %lu\n", (unsigned long)e);
    const QnnDevice_PlatformInfo_t *info = nullptr;
    e = q.deviceGetPlatformInfo(nullptr, &info);
    printf("deviceGetPlatformInfo: %lu\n", (unsigned long)e);
    if (e == QNN_SUCCESS && info && info->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
        for (uint32_t i = 0; i < info->v1.numHwDevices; i++) {
            const QnnDevice_HardwareDeviceInfo_t &d = info->v1.hwDevices[i];
            if (d.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;
            const QnnHtpDevice_DeviceInfoExtension_t *ext = (const QnnHtpDevice_DeviceInfoExtension_t *)d.v1.deviceInfoExtension;
            printf("hw device %u: id %u type %u cores %u", i, d.v1.deviceId, d.v1.deviceType, d.v1.numCores);
            if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP)
                printf(" | HTP on-chip: socModel %u, arch %u, dlbc %u, signed %u, vtcm %u MB", ext->onChipDevice.socModel,
                       ext->onChipDevice.arch, ext->onChipDevice.dlbcSupport, ext->onChipDevice.signedPdSupport, ext->onChipDevice.vtcmSize);
            printf("\n");
        }
        q.deviceFreePlatformInfo(nullptr, info);
    }
    return 0;
}
