import math

import cupy


def get_grid_thread_size(device: cupy.cuda.Device, number_of_item: int):
    max_block_dim_x = device.attributes["MaxGridDimX"]
    max_block_dim_y = device.attributes["MaxGridDimX"]
    max_grid_dim_x = device.attributes["MaxGridDimX"]
    max_grid_dim_y = device.attributes["MaxGridDimY"]

    max_threads_per_block = device.attributes["MaxThreadsPerBlock"]
    max_thread_size = min(math.floor(
        math.sqrt(max_threads_per_block)), number_of_item)

    if max_thread_size > max_block_dim_x or max_thread_size > max_block_dim_y:
        raise RuntimeError(
            "Cannot determine number of threads per block with simple algorithm. Dimensions are not equals or too many items or too weak device")

    grid_size, reminder = divmod(number_of_item, max_thread_size)

    if reminder > 0:
        grid_size += 1

    grid_size_xy = (grid_size, grid_size)

    if grid_size_xy[0] > max_grid_dim_x:
        raise RuntimeError("Grid size x limit is less than allowed")

    if grid_size_xy[1] > max_grid_dim_y:
        raise RuntimeError("Grid size y limit is less than allowed")

    if grid_size_xy[0] * max_thread_size < number_of_item:
        raise RuntimeError("Total evaluation units by X dim is less than number of items")

    if grid_size_xy[1] * max_thread_size < number_of_item:
        raise RuntimeError("Total evaluation units by Y dim is less than number of items")

    return grid_size_xy, (max_thread_size, max_thread_size)


ADJUSTED_COSINE_SIM_BETWEEN_ITEMS = cupy.RawKernel(r'''
extern "C" {

typedef unsigned long int_type;

typedef double rating_type;

typedef double rating_inter_type;

typedef struct  {
   rating_inter_type sum, correction;
} KahanSum;

__device__ void kahan_add(KahanSum * sum_state, rating_inter_type new_value)
{
   double y = new_value - sum_state->correction;
   double t = sum_state->sum + y;
   sum_state->correction = (t - sum_state->sum) - y;
   sum_state->sum = t;
}

__device__ int_type get_linear_index(int_type row, int_type col, int_type num_cols)
{
   return row * num_cols + col;
}

__device__ void assign_matrix_value(rating_type * matrix, int_type  row, int_type col, int_type num_cols, rating_type value)
{
   matrix[get_linear_index(row, col, num_cols)] = value;
}

__device__ rating_type get_matrix_value(const rating_type * matrix, int_type row, int_type col, int_type num_cols)
{
   return matrix[get_linear_index(row, col, num_cols)];
}

__global__ void adjusted_cosine_sim_between_items(
    int_type num_items,
    int_type min_estimations,
    const int_type * csc_col_pointers,
    const int_type * csc_row_indices,
    const rating_type * csc_values,
    rating_type * distance_matrix)
{
   // It is col number, actually
   int_type item_i_row = blockIdx.y * blockDim.y + threadIdx.y;
   int_type item_j_col = blockIdx.x * blockDim.x + threadIdx.x;

   if (item_i_row >= num_items || item_j_col >= num_items || item_j_col > item_i_row)
   {
      return;
   }

   if(item_j_col == item_i_row)
   {
      assign_matrix_value(distance_matrix, item_i_row, item_j_col, num_items, (rating_type)1.0);
      return;
   }

   int_type start_index_item_i = csc_col_pointers[item_i_row];
   const int_type end_index_item_i = csc_col_pointers[item_i_row + 1];

   int_type start_index_item_j = csc_col_pointers[item_j_col];
   const int_type end_index_item_j = csc_col_pointers[item_j_col + 1];

   // like merge sort find intersection between users. Row indices is sorted for CSC matrix representation
   KahanSum num { 0.0, 0.0}, norm_i {0.0, 0.0 }, norm_j {0.0, 0.0};
   rating_inter_type max_i = 1.0;
   rating_inter_type max_j = 1.0;
   bool is_first_max_i = true;
   bool is_first_max_j = true;
   int_type num_estimations = 0;

   while(start_index_item_i < end_index_item_i && start_index_item_j < end_index_item_j)
   {
      if(csc_row_indices[start_index_item_i] < csc_row_indices[start_index_item_j])
      {
         ++start_index_item_i;
      }
      // two items has estimation from one user
      else if(csc_row_indices[start_index_item_i] == csc_row_indices[start_index_item_j])
      {
         if(is_first_max_i)
         {
            max_i = csc_values[start_index_item_i];
            is_first_max_i = false;
         }
         else if(csc_values[start_index_item_i] > max_i)
         {
            max_i = csc_values[start_index_item_i];
         }

         if(is_first_max_j)
         {
            max_j = csc_values[start_index_item_j];
            is_first_max_j = false;
         }
         else if(csc_values[start_index_item_j] > max_j)
         {
            max_j = csc_values[start_index_item_j];
         }

         ++num_estimations;
         ++start_index_item_i;
         ++start_index_item_j;
      }
      else
      {
         ++start_index_item_j;
      }
   }

   if(num_estimations < min_estimations)
   {
      assign_matrix_value(distance_matrix, item_i_row, item_j_col, num_items, (rating_type)0.0);
      assign_matrix_value(distance_matrix, item_j_col, item_i_row, num_items, get_matrix_value(distance_matrix, item_i_row, item_j_col, num_items));
      return;
   }

   if(max_i == 0.0)
   {
      max_i = 1.0;
   }

   if(max_j == 0.0)
   {
      max_j = 1.0;
   }

   start_index_item_i = csc_col_pointers[item_i_row];
   start_index_item_j = csc_col_pointers[item_j_col];

   double term_norm_i, term_norm_j;

   while(start_index_item_i < end_index_item_i && start_index_item_j < end_index_item_j)
   {
      if(csc_row_indices[start_index_item_i] < csc_row_indices[start_index_item_j])
      {
         ++start_index_item_i;
      }
      // two items has estimation from one user
      else if(csc_row_indices[start_index_item_i] == csc_row_indices[start_index_item_j])
      {
         kahan_add(&num, csc_values[start_index_item_i] * csc_values[start_index_item_j]);
         term_norm_i = csc_values[start_index_item_i] / max_i;
         kahan_add(&norm_i, term_norm_i * term_norm_i);
         term_norm_j = csc_values[start_index_item_j] / max_j;
         kahan_add(&norm_j, term_norm_j * term_norm_j);
         ++start_index_item_i;
         ++start_index_item_j;
      }
      else
      {
         ++start_index_item_j;
      }
   }

   double inv_denum = rsqrt(norm_i.sum) / abs(max_i) * rsqrt(norm_j.sum) / abs(max_j);

   if(isinf(inv_denum))
   {
      inv_denum = 0.0;
   }

   assign_matrix_value(distance_matrix, item_i_row, item_j_col, num_items, (rating_type)(num.sum * inv_denum));
   assign_matrix_value(distance_matrix, item_j_col, item_i_row, num_items, get_matrix_value(distance_matrix, item_i_row, item_j_col, num_items));
}
};
''', "adjusted_cosine_sim_between_items", ("-prec-div=true", "-prec-sqrt=true"))
