import unittest
from split_recipe import assignments,development_groups

class SplitContracts(unittest.TestCase):
    def records(self):
        return [{'unit_id':f'synthetic-{subject}-{i}-{j}','metadata':{'subject':subject,'student_id':str(i)}} for subject in ['subject-a','subject-b'] for i in range(20) for j in range(2)]
    def test_order_independent(self):
        r=self.records()
        self.assertEqual(development_groups(r),development_groups(list(reversed(r))))
    def test_fifteen_percent_by_group(self):
        self.assertEqual(len(development_groups(self.records())),6)
    def test_groups_not_split(self):
        r=self.records();a=list(assignments(r));seen={}
        for row,result in zip(r,a):
            key=tuple(row['metadata'].values())
            self.assertEqual(seen.setdefault(key,result['split']),result['split'])
    def test_single_group_retained_for_train(self):
        self.assertEqual(development_groups(self.records()[:2]),set())

if __name__=='__main__':unittest.main()
