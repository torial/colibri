/* Worst-case operands for the integer dot kernels, at the contract of qrow_i8
 * (|x| <= 127, weights the full int8 range / every nibble). The AVX-512BW-without-VNNI
 * tier builds vpdpbusd out of vpmaddubsw + vpmaddwd, which saturates an int16 pair sum;
 * these inputs are the ones that would show it (all-max products of the same sign, in
 * every lane, at sizes that do and do not fill a vector). Every ISA tier must return the
 * plain-C value exactly.  Build: make tests/test_idot_extreme ARCH=skylake-avx512 */
#include "../idot.h"

static int fails=0;
static void chk(const char *what,int I,int64_t got,int64_t want){
    if(got!=want){ fprintf(stderr,"FAIL %s I=%d got %lld want %lld\n",what,I,(long long)got,(long long)want); fails++; }
}
int main(void){
    static const int sizes[]={1,15,16,31,32,63,64,65,127,128,129,255,256,1000,4096,4097};
    static const int8_t wv[]={-128,127,-127,1,0};
    static const int8_t xv[]={127,-127,1,-1,0};
    int8_t *w=malloc(8192),*x=malloc(4*8192); uint8_t *w4=malloc(8192);
    for(unsigned si=0;si<sizeof sizes/sizeof *sizes;si++){
        int I=sizes[si];
        for(unsigned a=0;a<sizeof wv;a++) for(unsigned b=0;b<sizeof xv;b++){
            for(int i=0;i<I;i++){ w[i]=wv[a]; x[i]=xv[b]; }
            int64_t want=0; for(int i=0;i<I;i++) want+=(int32_t)w[i]*x[i];
            chk("dot_i8i8",I,dot_i8i8(w,x,I),want);
            int32_t out[4];
            for(int r=1;r<=4;r++){
                for(int rr=0;rr<r;rr++) for(int i=0;i<I;i++) x[(size_t)rr*I+i]=xv[b];
                dot_i8i8_rows(w,x,I,r,out);
                for(int rr=0;rr<r;rr++) chk("dot_i8i8_rows",I,out[rr],want);
            }
            for(int i=0;i<I;i++) x[i]=xv[b];
        }
        /* int4: every nibble value against the extreme activations */
        for(int nib=0;nib<16;nib++) for(unsigned b=0;b<sizeof xv;b++){
            for(int i=0;i<(I+1)/2;i++) w4[i]=(uint8_t)(nib|(nib<<4));
            for(int i=0;i<I;i++) x[i]=xv[b];
            int64_t wu=0,ws=0; for(int i=0;i<I;i++){ wu+=(int32_t)nib*x[i]; ws+=(int32_t)(nib-8)*x[i]; }
            chk("dot_i4i8",I,dot_i4i8(w4,x,I),ws);
            /* dot_i4p_u reads the planar layout; planar nibbles all equal -> any order gives wu
             * for the whole 64-blocks, and the unsigned tail pairs follow the same identity */
            if(I%64==0) chk("dot_i4p_u",I,dot_i4p_u(w4,x,I),wu);
        }
    }
    printf("idot extreme operands (%s): %s\n",IDOT_KERNEL,fails?"FAILED":"ok");
    return fails?1:0;
}
