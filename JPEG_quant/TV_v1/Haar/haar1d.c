

/* Inputs: vec = input vector, n = size of input vector */
void haar1d(float *vec, int n)
{
     int i=0;
     int w=n;
     float *vecp = new float[n];
     for(i=0;i<n;i++)
          vecp[i] = 0;

     while(w>1)
     {
          w/=2;
          for(i=0;i<w;i++)
          {
               vecp[i] = (vec[2*i] + vec[2*i+1])/sqrt(2.0);
               vecp[i+w] = (vec[2*i] - vec[2*i+1])/sqrt(2.0);
          }

          for(i=0;i<(w*2);i++)
               vec[i] = vecp[i]; 
     }

     delete [] vecp;
}
 