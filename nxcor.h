void icvcalculatebuffersizes(int mi,int ni,int mt,int nt,
      int *imgbufsize, int *templbufsize, int *sumbufsize,
      int * sqsumbufsize, int*resnumbufsize, int*resdenombufsize);
void icvmatchtemplateentry(int mi,int ni,int mt,int nt,
    const float* image, const float* _template, float*buffer,
    float **imgbuf, float ** templbuf,
    float **sumbuf, float **sqsumbuf, float **resnum, float **resdenom);
float icvcrosscorr(const float *vec1, const float *vec2, int len );
float icvsumpixels(const float *vec, int len);
void icvmatchtemplate17(float*image, float*_template, 
    float*result,int mi,int ni,int mt,int nt,void*buffer);
void cvmatchtemplate(float* image,float* _template,float*result,int mi,int ni,int mt,int nt, void*buffer);
void nxcor(float*image, float*_template, float*result, int nm, int mi, int ni, int mt, int nt);
void icvmatchtemplategetbufsize_coeffnormed(int mi,int ni,int mt,int nt,size_t *buffersize);
