"""PK-сэмплер: в батче P личностей по K кадров, кадры одной личности — с разных камер.

Разные камеры в батче нужны, чтобы triplet-лосс тянул вместе именно кросс-камерные пары:
внутрикамерные пары модель и так сближает (docs/findings.md §6).

Дополнительно кадры одной личности разводятся по ракурсу. Разбор ошибок фузии (24.09) показал
главный сценарий провала: модель ставит выше «тот же ракурс, другая машина», чем «другой ракурс,
та же машина» — Mercedes спереди проигрывает ему же сзади. Если четыре кадра личности в батче сняты
с похожего ракурса, triplet этот трудный случай просто не видит.

Ракурс приближается логарифмом отношения сторон BBox: анфас и корма узкие, борт широкий. Это
входные данные, а не `camera_id`, так что правило 2 не затрагивается.

Батч можно набирать не случайными личностями, а **путающимися**: одна машина плюс её ближайшие
соседи по эмбеддингу (`scripts/mine_hard_ids.py`). Замер 24.09: выше правильного ответа стоят
другие автомобили той же модели и цвета, и 91,5% из них сняты другими камерами — значит дело не в
ракурсе, а в том, что модель их не различает. При 16 случайных личностях из 1232 два одинаковых
каршеринговых автомобиля попадают в один батч примерно раз в сто эпох, и triplet-лосс этой пары
просто не видит. Часть батча остаётся случайной: классификационному лоссу нужны разные машины.
"""
from __future__ import annotations

import collections
import random

from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, labels: list[int], cams: list[int], p: int, k: int, seed: int = 0,
                 views: list[float] | None = None,
                 neighbors: dict[int, list[int]] | None = None, hard_frac: float = 0.0):
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.views = views
        self.neighbors, self.hard_frac = neighbors, hard_frac
        self.by_label: dict[int, dict[int, list[int]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for i, (lab, cam) in enumerate(zip(labels, cams)):
            self.by_label[lab][cam].append(i)
        self.labels = sorted(self.by_label)
        self.n_batches = max(1, len(labels) // (p * k))

    def __len__(self) -> int:
        return self.n_batches * self.p * self.k

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _pick(self, pool: list[int], chosen: list[int]) -> int:
        """Кадр из пула: без ракурсов — случайный (пул уже перемешан), иначе самый непохожий
        по ракурсу на уже взятые."""
        if self.views is None or not chosen:
            return pool[-1]
        return max(pool, key=lambda j: min(abs(self.views[j] - self.views[c]) for c in chosen))

    def _take(self, rng: random.Random, label: int) -> list[int]:
        """K кадров одной личности: сначала по одному с разных камер, потом добор."""
        cams = list(self.by_label[label])
        rng.shuffle(cams)
        pools = {c: rng.sample(self.by_label[label][c], len(self.by_label[label][c])) for c in cams}
        out = []
        while len(out) < self.k:
            progressed = False
            for c in cams:
                if pools[c]:
                    idx = self._pick(pools[c], out)
                    pools[c].remove(idx)
                    out.append(idx)
                    progressed = True
                    if len(out) == self.k:
                        break
            if not progressed:  # кадров меньше K — повторяем с начала
                pools = {c: rng.sample(self.by_label[label][c], len(self.by_label[label][c])) for c in cams}
        return out

    def _batch_labels(self, rng: random.Random, seed_label: int, pool: list[int]) -> list[int]:
        """P личностей батча: якорь, его путающиеся соседи и случайный добор."""
        chosen = [seed_label]
        if self.neighbors and self.hard_frac:
            near = [v for v in self.neighbors.get(seed_label, []) if v in self.by_label]
            want = int(round((self.p - 1) * self.hard_frac))
            # Берём не подряд первых, а случайных из ближних: иначе каждая эпоха даёт один и тот
            # же батч вокруг машины и модель переобучается на конкретную пару
            take = near[: max(want * 2, want)]
            chosen += rng.sample(take, min(want, len(take)))
        for label in pool:
            if len(chosen) >= self.p:
                break
            if label not in chosen:
                chosen.append(label)
        return chosen[: self.p]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        labels = self.labels[:]
        rng.shuffle(labels)
        order, pos = [], 0
        for _ in range(self.n_batches):
            if pos + self.p > len(labels):
                rng.shuffle(labels)
                pos = 0
            batch = self._batch_labels(rng, labels[pos], labels[pos + 1:] + labels[:pos])
            for label in batch:
                order.extend(self._take(rng, label))
            pos += self.p
        return iter(order)
