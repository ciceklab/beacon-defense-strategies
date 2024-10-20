import torch


class MAFHelper():
    mafs: torch.Tensor
    maf_categories: torch.Tensor
    cat_start_ind: torch.Tensor

    def __init__(self, mafs, num_groups=None) -> None:
        self.mafs = mafs
        self.cat_start_ind = torch.zeros(
            6, device=mafs.device, dtype=torch.long)

        if num_groups != None:
            self.num_groups = num_groups
            # self.group_boundaries = self._get_bouderies()
            self.maf_categories = self._get_categorized_mafs()

    def get_mafs(self) -> torch.Tensor:
        return self.mafs.clone()

    def get_mafs_categories(self) -> torch.Tensor:
        return self.maf_categories.clone()

    # def get_category(self, maf):
    #     for i in range(0, len(self.group_boundaries - 1)):
    #         if self.group_boundaries[i] <= maf < self.group_boundaries[i + 1]:
    #             return i

    #     return len(self.group_boundaries) - 1

    def get_category(self, maf) -> int:
        if maf < .03:
            return 0
        elif maf < 0.1:
            return 1
        elif maf < 0.2:
            return 2
        elif maf < 0.3:
            return 3
        elif maf < 0.4:
            return 4
        else:
            return 5

    # def _get_bouderies(self):
    #     sorted_numbers = torch.sort(self.mafs).values

    #     n = len(sorted_numbers)
    #     split_indices = [int(n * i / self.num_groups)
    #                      for i in range(1, self.num_groups)]
    #     group_boundaries = [sorted_numbers[0].item()] + [sorted_numbers[idx].item()
    #                                                      for idx in split_indices] + [sorted_numbers[-1].item()]
    #     group_boundaries = [round(bound, 3) if bound >
    #                         1e-6 else 0.0 for bound in group_boundaries]

    #     return group_boundaries

    def _get_categorized_mafs(self):
        maf_categories = torch.zeros_like(
            self.mafs, device=self.mafs.device, dtype=torch.int16)

        new_cat = None
        for idx, maf_value in enumerate(self.mafs):
            category = self.get_category(maf_value)
            maf_categories[idx] = category

            if new_cat != category:
                new_cat = category
                self.cat_start_ind[category] = idx


        return maf_categories
