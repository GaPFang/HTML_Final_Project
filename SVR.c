#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

int main() {
    double C[5] = {0.01, 0.1, 1, 10, 100};
    double gamma[5] = {0.01, 0.1, 1, 10, 100};
    double epsilon[5] = {0.01, 0.1, 1, 10, 100};

    int fds[5][5][5][2];

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                pipe(fds[i][j][k]);
                if (fork() == 0) {
                    dup2(fds[i][j][k][1], 1);
                    close(fds[i][j][k][0]);
                    close(fds[i][j][k][1]);
                    execlp("python3", "python3", "SVR.py", C[i], gamma[j], epsilon[k], NULL);
                }
                close(fds[i][j][k][1]);
            }
        }
    }
    int total = 125;
    while(total--) {
        int status;
        wait(&status);
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                char result[100];
                read(fds[i][j][k][0], result, 100);
                printf("%s\n", result);
            }
        }
    }

    return 0;
}