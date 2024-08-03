#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

void copy_file(const char *sourcePath, const char *destinationPath) {
    FILE *sourceFile, *destinationFile;
    char buffer[BUFSIZ];
    size_t n;

    // Open the source file in binary read mode
    sourceFile = fopen(sourcePath, "rb");
    if (sourceFile == NULL) {
        perror("Error opening source file");
        exit(EXIT_FAILURE);
    }

    // Open the destination file in binary write mode
    destinationFile = fopen(destinationPath, "wb");
    if (destinationFile == NULL) {
        perror("Error opening destination file");
        fclose(sourceFile);
        exit(EXIT_FAILURE);
    }

    // Copy the contents of the source file to the destination file
    while ((n = fread(buffer, 1, sizeof(buffer), sourceFile)) > 0) {
        if (fwrite(buffer, 1, n, destinationFile) != n) {
            perror("Error writing to destination file");
            fclose(sourceFile);
            fclose(destinationFile);
            exit(EXIT_FAILURE);
        }
    }

    // Close the files
    fclose(sourceFile);
    fclose(destinationFile);
}

void copy_dir(const char *sourceDir, const char *destDir) {
    DIR *dir;
    struct dirent *entry;

    dir = opendir(sourceDir);
    if (!dir) {
        perror("Error opening source directory");
        exit(EXIT_FAILURE);
    }

    if (mkdir(destDir, 0777) == -1 && errno != EEXIST) {
        perror("Error creating destination directory");
        closedir(dir);
        exit(EXIT_FAILURE);
    }

    // Read entries in the source directory
    while ((entry = readdir(dir)) != NULL) {
        char sourcePath[PATH_MAX];
        char destPath[PATH_MAX];

        // Skip "." and ".." entries
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        snprintf(sourcePath, sizeof(sourcePath), "%s/%s", sourceDir, entry->d_name);
        snprintf(destPath, sizeof(destPath), "%s/%s", destDir, entry->d_name);

        struct stat entryInfo;
        if (stat(sourcePath, &entryInfo) == -1) {
            perror("Error getting entry information");
            closedir(dir);
            exit(EXIT_FAILURE);
        }

        if (S_ISDIR(entryInfo.st_mode)) { copyDirectory(sourcePath, destPath); }  
        else { copyFile(sourcePath, destPath); }
    }

    closedir(dir);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source directory> <destination directory>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    copyDirectory(argv[1], argv[2]);

    printf("Directory copied successfully.\n");

    return 0;
}
