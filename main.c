/*
  compile with :
  gcc -Wall -march=native -O3 -I${FFTWDIR}/include ssdisk.c -L${FFTWDIR}/lib -lfftw3 -lm -o ssdisk
*/

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <dirent.h>
#include <time.h>

/***************** basic utilities *****************/
 

FILE *fopen_check(char *filename, char *mode)
{
  FILE *fp = fopen(filename, mode);
  if (!fp) {
    fprintf(stderr, "Error opening file %s with mode %s\n", filename, mode);
    perror("fopen"); exit(EXIT_FAILURE);
  }
  return fp;
}

size_t fread_check(void *p, size_t size, size_t n, FILE *fp)
{
  if (fread(p, size, n, fp) != n) {
    fprintf(stderr, "Error reading %ld items of size %ld\n",
            (long)n, (long)size);
    perror("fread"); exit(EXIT_FAILURE);
  }
  return n;
}

size_t fwrite_check(void *p, size_t size, size_t n, FILE *fp)
{
  if (fwrite(p, size, n, fp) != n) {
    fprintf(stderr, "Error writing %ld items of size %ld\n",
            (long)n, (long)size);
    perror("fwrite"); exit(EXIT_FAILURE);
  }
  return n;
}

void *malloc_check(size_t s)
{
  void *p=malloc(s);
  if (!p) {
    fprintf(stderr, "Error allocating %ld bytes: ", s);
    perror("malloc"); exit(EXIT_FAILURE);
  }
  return p;
}

void *calloc_check(size_t n, size_t s)
{
  void *p=calloc(n,s);
  if (!p) {
    fprintf(stderr, "Error allocating %ld elements of %ld bytes: ",n,s);
    perror("calloc"); exit(EXIT_FAILURE);
  }
  return p;
}

double **matrix(size_t nrow, size_t ncol)
/* allocate a matrix with subscript range m[0..nrow-1][0..ncol-1] */
{
  size_t i;
  double **m;
//if x is a pointer, both x[0]==*(x+0) and *x is the value of the referred variable.

  m=(double **)malloc_check((size_t)(nrow*sizeof(double*))); // m is a 1-D array of pointers that points to the first element of another array of pointers *m
  *m=(double *)calloc_check(nrow*ncol,sizeof(double)); //*m is a 1-D array of pointers, of the size of the whole matrix nrow*ncol. so the first element of *m points to the [1,1] element of the matrix, the second to [2,1], and the ncol-th to [1,2]
  for(i=1;i<nrow;i++) m[i]=m[i-1]+ncol; //m[0] is a pointer to the first element of the matrix. m[1]=m[0]+ncols is the pointer to the pointer of the [2,1] element, m[2]=m[0]+2ncol points to [3,1], etc. (recall pointer+1 points to the next double in memory)
  return m;
}

double **aligned_matrix(size_t nrow, size_t ncol)
/* allocate a matrix with subscript range m[0..nrow-1][0..ncol-1] */
{
  size_t i;
  double **m;
  m=(double **)malloc_check((size_t)(nrow*sizeof(double*)));
  *m=(double *)fftw_malloc(nrow*ncol*sizeof(double));
  if (!m) {
    fprintf(stderr, "Error: unable to allocate %ld*%ld matrix\n",nrow,ncol);
    perror("fftw_malloc"); exit(EXIT_FAILURE);
  }
  memset(*m, 0, nrow*ncol*sizeof(double));
  for(i=1;i<nrow;i++) m[i]=m[i-1]+ncol;
  return m;
}

double complex **aligned_cmatrix(size_t nrow, size_t ncol)
/* allocate a matrix with subscript range m[0..nrow-1][0..ncol-1] */
{
  size_t i, nc=ncol/2+1;
  double complex **m;

  m=(double complex **)malloc_check(nrow*sizeof(double complex *));
  *m=(double complex *)fftw_malloc(nrow*nc*sizeof(double complex));
  if (!m) {
    fprintf(stderr, "Error: unable to allocate %ld*%ld cmatrix\n",nrow,ncol);
    perror("fftw_malloc"); exit(EXIT_FAILURE);
  }
  memset(*m, 0, nrow*nc*sizeof(double complex));
  for(i=1;i<nrow;i++) m[i]=m[i-1]+nc;
  return m;
}

int get_option(int argc, char *argv[], char format[],...)
{
  va_list ap;
  char optionstr[100];
  char *tmp;
  int i;
 
  strcpy(optionstr,format); /* optionstr points at the option */
  tmp=strpbrk(optionstr," \t\n");
  if (tmp!=NULL) {
    tmp[0]='\0';
    tmp++;
    tmp=strchr(tmp,'%');              /* tmp points at the format string */
  }
  /*  printf("|%s| |%s|\n",optionstr,tmp); */
  i=0;
  while ((i<argc)&&(strcmp(argv[i],optionstr))) i++;
  if (i==argc) return 0;                  /* option doesn't exist */
  if ((i<argc) && (tmp==NULL)) return 1;  /* option found and not expected */
                                          /* to have arguments */
  /* advance to the first value to be read */
  optionstr[0]='*';                       /* null out the option */
  va_start(ap,format); /* start with first variable address */
 
  ++i;
  /*  printf("|%s| |%s|\n",argv[i],tmp);*/
  while ((i<argc) &&
         (tmp!=NULL) &&
         ((argv[i][0]!='-') || (isdigit(argv[i][1]))))
    {
  
/*       printf("|%s| |%s|\n",argv[i],tmp); */
      switch (tmp[1]) {
 
      case 'd':
        sscanf(argv[i],"%d",(va_arg(ap,int *)));
        break;
      case 'f':
        sscanf(argv[i],"%f",(va_arg(ap,float *)));
        break;
      case 's':
        sscanf(argv[i],"%s",(va_arg(ap,char *)));
        break;
      case 'l':
        if (tmp[2] == 'f') sscanf(argv[i],"%lf",(va_arg(ap,double *)));
        else if (tmp[2] == 'd') sscanf(argv[i],"%ld",(va_arg(ap,long *)));
        else return -1;
        break;
      default: return -1; break;
      }
 
      i++;
      tmp++;
      tmp=strchr(tmp,'%');
    }
 
  va_end(ap);
 
  if (i==argc) return -1;
 
  return 1;
}



double **zeropad(int n, double **orig)
{
  int i, n_2 = n/2;
  double **padded = aligned_matrix(2*n, 2*n);
  for (i=0; i<n; ++i) memcpy(padded[n_2+i]+n_2, orig[i], n*sizeof(double));
  return padded;
}

double sqr(double x) {return x*x;}
double cube(double x) {return x*x*x;}
double csq(double complex z)
{
  double r = creal(z), i=cimag(z);
  return r*r + i*i;
}


/***************** Shakura-Sunyaev disk *****************/


// disk parameters

double inc=0, sini=0, cosi=1, phi0=0;
double Rg=1, R0=50, Rin=6, slope=3;

double f_SS(double R, double Doppler)
{
  return R>Rin ? Doppler*pow(pow(R0/R,slope)*(1-sqrt(Rin/R)),-0.25) : 0;
}

double I_SS(double R, double Doppler)
{
  return R>Rin ? cube(Doppler)/(exp(f_SS(R,Doppler))-1) : 0;
}


float randomFloat(float x) {
return ((float)rand() / RAND_MAX) * 2.0f * x - x;
}

int main(int argc, char *argv[])
{
    
    
  int npix = 64, n2, i, j;
  double xp, yp, cp, sp;
  double Rmax, dx, x, y, R, v=0, Doppler=1;
  double **data=NULL, **v2data, **vargdata, sum;
  double complex **FT;
  fftw_plan plan;
  char USE_BEAM;
    
  char *run;
  run = (char *)malloc(50 * sizeof(char));
  int cluster=0;
  double noiselevel=0;
  int randomseedint;
  unsigned int randomseed;
  double noisegen;



  if (get_option(argc, argv, "-h") || get_option(argc, argv, "--help")) {
    fputs("Usage: ssdisk [options]\n", stderr);
    fputs("\t-R0 R0 [50.0]\n", stderr);
    fputs("\t-Rg Rg [1.0]\n", stderr);
    fputs("\t-Rin Rin [6.0]\n", stderr);
    fputs("\t-Rmax Rmasx [150.0]\n", stderr);
    fputs("\t-n slope [3.0]\n", stderr);
    fputs("\t-i inclination [0]\n", stderr);
    fputs("\t-phi positionangle [0]\n", stderr);
    fputs("\t-np num_pixels_per_side [512]\n", stderr);
    return 0;
  }

  USE_BEAM = !get_option(argc, argv, "-nobeam");
  get_option(argc, argv, "-R0 %lf", &R0);
  get_option(argc, argv, "-Rin %lf", &Rin);
  get_option(argc, argv, "-Rg %lf", &Rg);
  get_option(argc, argv, "-n %lf", &slope);
  get_option(argc, argv, "-phi %lf", &phi0);
  get_option(argc, argv, "-run %s", run);
  get_option(argc, argv, "-cluster %d", &cluster); //0 is local, 1 is cluster
  get_option(argc, argv, "-cluster %d", &cluster); //0 is local, 1 is cluster
  get_option(argc, argv, "-noise %lf", &noiselevel);
  get_option(argc, argv, "-seed %d", &randomseedint);

    randomseed=(unsigned int)randomseedint;
//  printf("noise %lf \n",noiselevel);
    FILE *textintensity;
    FILE *textvisibility;
    FILE *textarg;
    
    if(cluster==1){
        char *textfilename;
        textfilename = (char *)malloc(256*sizeof(char));
        strcpy(textfilename,"files/outputint/intensityfile");
        strcat(textfilename,run);
        //  printf("%s \n",textfilename);
        textintensity = fopen(textfilename, "w");
        
        strcpy(textfilename, "files/outputvis/visibilityfile");
        strcat(textfilename,run);
        // printf("%s \n",textfilename);
        textvisibility = fopen(textfilename, "w");
        
        strcpy(textfilename, "files/outputarg/argfile");
        strcat(textfilename,run);
        // printf("%s \n",textfilename);
        textarg = fopen(textfilename, "w");
    } else {
        char *textfilename;
        textfilename = (char *)malloc(256*sizeof(char));
        strcpy(textfilename,"/Users/danielegana/files/outputint/intensityfile");
        strcat(textfilename,run);
        //  printf("%s \n",textfilename);
        textintensity = fopen(textfilename, "w");
        
        strcpy(textfilename, "/Users/danielegana/files/outputvis/visibilityfile");
        strcat(textfilename,run);
        // printf("%s \n",textfilename);
        textvisibility = fopen(textfilename, "w");
        
        strcpy(textfilename, "/Users/danielegana/files/outputarg/argfile");
        strcat(textfilename,run);
        // printf("%s \n",textfilename);
        textarg = fopen(textfilename, "w");
    }


  if (get_option(argc, argv, "-i %lf", &inc)) {
    cosi = cos(inc);
    sini = sin(inc);
  }
  get_option(argc, argv, "-np %d", &npix);
  Rmax = 3*R0;
  get_option(argc, argv, "-Rmax %lf", &Rmax);
  cp = cos(phi0);
  sp = sin(phi0);

  if (Rin <= Rg) {
    fprintf(stderr, "Error: Rin=%g is less than Rg=%g\n", Rin, Rg);
    exit(EXIT_FAILURE);
  }


  dx = 2*Rmax/npix;
  data = aligned_matrix(npix, npix);
  n2 = npix*npix;

  for (i=0; i<npix; ++i) {
    xp = -Rmax + (i+0.5)*dx;
    for (j=0; j<npix; ++j) {
      yp = -Rmax + (j+0.5)*dx;

      x = cp*xp - sp*yp; // xp,yp are the coordinates in the observed plane. x,y in the one where the disk is round. So to go back to round I first rotate the inclined disk back to the horizontal position, and then I remove the inclination.
      y = sp*xp + cp*yp;
      x /= cosi;
      R = hypot(x,y);
      if (USE_BEAM && R>Rg) {
    v = sqrt(Rg/R);    // note v<1 since R>Rg
    Doppler = sqrt(1-v*v)/(1-v*sini*y/R);
      } else Doppler=1;
      
      data[i][j] = I_SS(R, Doppler);
    }
  }

  // renormalize to sum=1
  for (sum=0, i=0; i<n2; ++i) sum += (*data)[i]; //See double pointer explanations aboce
  if (sum<=0) {
    fprintf(stderr, "Error: sum(pixels)=%g\n", sum);
    exit(EXIT_FAILURE);
  }
  for (i=0; i<n2; ++i) (*data)[i] /= sum;

    srand(randomseed);


  FT = aligned_cmatrix(npix, npix);
  plan = fftw_plan_dft_r2c_2d(npix, npix, *data, *FT, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  v2data = matrix(npix, npix/2 + 1);
    for (i=0; i<npix; ++i) for (j=0; j<=npix/2; ++j) {
        noisegen=randomFloat(noiselevel);
       // printf("test noise %lf\n",noisegen);
        v2data[i][j] = csq(FT[i][j])+noisegen;}
  
  vargdata = matrix(npix, npix/2 + 1);
      for (i=0; i<npix; ++i) for (j=0; j<=npix/2; ++j) vargdata[i][j] = carg(FT[i][j]);

      
 
    
    // Write tables to text data 
    for (int i = 0; i < npix; i++) {
        for (int j = 0; j < npix; j++) {
            fprintf(textintensity, "%f ", data[i][j]);
        }
        fprintf(textintensity, "\n"); // Move to the next line after each row
    }
    

    for (i=0; i<npix; ++i) {
        for (j=0; j<=npix/2; ++j) {
            fprintf(textvisibility, "%f ", v2data[i][j]);
            }
        fprintf(textvisibility, "\n"); // Move to the next line after each row
    }

    
    for (i=0; i<npix; ++i) {
        for (j=0; j<=npix/2; ++j) {
            fprintf(textarg, "%f ", vargdata[i][j]);
            }
       fprintf(textarg, "\n"); // Move to the next line after each row
    }

  

  fftw_free(*data); free(data);
  fftw_free(*FT); free(FT);
  free(*v2data); free(v2data);

  return 0;
}


