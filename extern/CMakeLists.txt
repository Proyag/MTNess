set(SPM_ENABLE_TCMALLOC ON CACHE BOOL "Enable TCMalloc if available." FORCE)
set(SPM_ENABLE_SHARED ON CACHE BOOL "Builds shared libaries in addition to static libraries." FORCE)
set(SPM_TCMALLOC_STATIC OFF CACHE BOOL "Link static library of TCMALLOC." FORCE)

add_subdirectory(sentencepiece)

set_target_properties(spm_encode spm_decode spm_train spm_normalize spm_export_vocab
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
