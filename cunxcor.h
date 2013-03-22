#ifdef __cplusplus
extern "C"
{
#endif
void cunxcor(float*image, float*Template, float*result, 
    int nm, int mi, int ni, int mt, int nt);
void nxcor_sse(float*image, float*Template, float*result, 
    int nm, int mi, int ni, int mt, int nt);
void nxcor_avx(float*image, float*Template, float*result, 
    int nm, int mi, int ni, int mt, int nt);
#ifdef __cplusplus
}
#endif

