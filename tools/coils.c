// write the three Modbus RTU coils of the light outputs: coils <device> <c1> <c2> <c3>
#include <modbus/modbus.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: coils <device> <c1> <c2> <c3>\n"); return 2; }
    modbus_t *b = modbus_new_rtu(argv[1], 9600, 'N', 8, 1); if (!b) { perror("modbus_new_rtu"); return 1; }
    modbus_set_slave(b, 1); modbus_set_response_timeout(b, 1, 0);
    if (modbus_connect(b) < 0) { fprintf(stderr, "connect: %s\n", modbus_strerror(errno)); return 1; }
    uint8_t v[3] = { (uint8_t)atoi(argv[2]), (uint8_t)atoi(argv[3]), (uint8_t)atoi(argv[4]) };
    int n = modbus_write_bits(b, 0, 3, v);
    printf(n == 3 ? "DO=%d%d%d written\n" : "write failed: %s\n", n == 3 ? v[0] : 0, v[1], v[2]);
    if (n != 3) fprintf(stderr, "%s\n", modbus_strerror(errno));
    modbus_close(b); modbus_free(b); return n == 3 ? 0 : 1;
}
