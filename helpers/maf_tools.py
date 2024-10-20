import torch


class MAFHelper():
    mafs: torch.Tensor
    maf_categories: torch.Tensor

    def __init__(self, mafs, num_groups=None) -> None:
        self.mafs = mafs

        if num_groups != None:
            self.num_groups = num_groups
            self.group_boundaries = self._get_bouderies()

    def get_mafs(self) -> torch.Tensor:
        return self.mafs.clone()

    def get_mafs_categories(self) -> torch.Tensor:
        return self.maf_categories.clone()

    def get_category(self, maf):
        for i in range(1, len(self.group_boundaries)):
            if self.group_boundaries[i] <= maf < self.group_boundaries[i + 1]:
                return i

        return len(self.group_boundaries) - 1

    def _get_bouderies(self):
        sorted_numbers = torch.sort(self.mafs).values

        n = len(sorted_numbers)
        split_indices = [int(n * i / self.num_groups)
                         for i in range(1, self.num_groups)]
        group_boundaries = [sorted_numbers[0].item()] + [sorted_numbers[idx].item()
                                                         for idx in split_indices] + [sorted_numbers[-1].item()]
        group_boundaries = [round(bound, 3) if bound >
                            1e-6 else 0.0 for bound in group_boundaries]

        return group_boundaries

    def _get_categorized_mafs(self):
        maf_categories = torch.zeros_like(
            self.mafs, device=self.mafs.device, dtype=torch.int16)

        for idx, maf_value in enumerate(self.mafs):
            maf_categories[idx] = self.get_category(
                maf_value, self.group_boundaries)

        return maf_categories
