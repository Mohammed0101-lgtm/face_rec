//
// automation code for dataset loading to workspace directory
//

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

void copy_file(const char *source, const char *dest) {
    FILE*  source, *dest;
    char   buffer[BUFSIZ];
    size_t n;

    // Open the source file in binary read mode
    source = fopen(source, "rb");
    if (!source) {
        perror("Error opening source file");
        exit(EXIT_FAILURE);
    }

    // Open the destination file in binary write mode
    dest = fopen(dest, "wb");
    if (!dest) {
        perror("Error opening destination file");
        fclose(source);
        exit(EXIT_FAILURE);
    }

    // Copy the contents of the source file to the destination file
    while ((n = fread(buffer, 1, sizeof(buffer), source)) > 0) 
        if (fwrite(buffer, 1, n, dest) != n) {
            perror("Error writing to destination file");
            fclose(source);
            fclose(destinationFile);
            exit(EXIT_FAILURE);
        }

    // Close the files
    fclose(source);
    fclose(dest);
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
        char source[PATH_MAX];
        char dest[PATH_MAX];

        // Skip "." and ".." entries
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) 
            continue;
        

        snprintf(source, sizeof(source), "%s/%s", sourceDir, entry->d_name);
        snprintf(dest, sizeof(dest), "%s/%s", destDir, entry->d_name);

        struct stat entryInfo;
        if (stat(source, &entryInfo) == -1) {
            perror("Error getting entry information");
            closedir(dir);
            exit(EXIT_FAILURE);
        }

        if (S_ISDIR(entryInfo.st_mode)) 
            copyDirectory(source, dest);   
        else  
            copyFile(source, dest); 
    }

    closedir(dir);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, 
                "Usage: %s <source directory> <destination directory>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    copyDirectory(argv[1], argv[2]);

    printf("Directory copied successfully.\n");

    return 0;
}
