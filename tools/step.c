// pulse an A4988 from a Linux gpiochip (v2 ABI): step <chip> <dir_line> <step_line> <steps> <half_us> [dir]
#include <fcntl.h>
#include <linux/gpio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: step <chip> <dir_line> <step_line> <steps> <half_us> [dir 0|1]\n"); return 2; }
    int fd = open(argv[1], O_RDONLY | O_CLOEXEC); if (fd < 0) { perror("open chip"); return 1; }
    struct gpio_v2_line_request req; memset(&req, 0, sizeof req);
    req.offsets[0] = atoi(argv[2]); req.offsets[1] = atoi(argv[3]); req.num_lines = 2;
    req.config.flags = GPIO_V2_LINE_FLAG_OUTPUT; strncpy(req.consumer, "step", sizeof req.consumer - 1);
    if (ioctl(fd, GPIO_V2_GET_LINE_IOCTL, &req) < 0) { perror("get lines"); return 1; }
    int steps = atoi(argv[4]), half = atoi(argv[5]), dir = argc > 6 ? atoi(argv[6]) : 0;
    struct gpio_v2_line_values v; v.mask = 3; v.bits = dir ? 1 : 0;
    ioctl(req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v);
    for (int i = 0; i < steps; i++) {
        v.bits = (dir ? 1 : 0) | 2; ioctl(req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v); usleep(half);
        v.bits = (dir ? 1 : 0);     ioctl(req.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &v); usleep(half);
    }
    printf("%d steps, half-cycle %d us, dir %d\n", steps, half, dir);
    return 0;
}
