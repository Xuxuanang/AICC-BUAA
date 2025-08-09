import API
import mindspore

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class PreNorm(mindspore.nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = mindspore.nn.LayerNorm(dim)
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(mindspore.nn.Cell):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = mindspore.nn.SequentialCell(mindspore.nn.Dense(in_channels=dim, out_channels=hidden_dim, weight_init="normal", bias_init="zeros", has_bias=True, activation=None), mindspore.nn.GELU(approximate=False), mindspore.nn.Dropout(keep_prob=dropout, dtype="mindspore.float32"), mindspore.nn.Dense(in_channels=hidden_dim, out_channels=dim, weight_init="normal", bias_init="zeros", has_bias=True, activation=None), mindspore.nn.Dropout(keep_prob=dropout, dtype="mindspore.float32"))

    def construct(self, x):
        return self.net(x)


class Attention(mindspore.nn.Cell):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = mindspore.nn.Softmax(axis=-1)
        self.dropout = mindspore.nn.Dropout(keep_prob=dropout, dtype="mindspore.float32")

        self.to_qkv = mindspore.nn.Dense(in_channels=dim, out_channels=inner_dim * 3, weight_init="normal", bias_init="zeros", has_bias=False, activation=None)

        self.to_out = mindspore.nn.SequentialCell(mindspore.nn.Dense(inner_dim, dim),mindspore.nn.Dropout(dropout)) if project_out else mindspore.nn.Identity()

    def construct(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = mindspore.ops.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = mindspore.ops.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(mindspore.nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = mindspore.nn.CellList([])
        for _ in range(depth):
            self.layers.append(mindspore.nn.CellList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(mindspore.nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = mindspore.nn.SequentialCell(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),mindspore.nn.LayerNorm((patch_dim,)),mindspore.nn.Dense(patch_dim, dim),mindspore.nn.LayerNorm((dim,)),)

        self.pos_embedding = mindspore.Parameter(mindspore.ops.randn(1, num_patches + 1, dim))
        self.cls_token = mindspore.Parameter(mindspore.ops.randn(1, 1, dim))
        self.dropout = mindspore.nn.Dropout(keep_prob=emb_dropout, dtype="mindspore.float32")

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = mindspore.nn.Identity()

        self.mlp_head = mindspore.nn.SequentialCell(mindspore.nn.LayerNorm(dim), mindspore.nn.Dense(in_channels=dim, out_channels=num_classes, weight_init="normal", bias_init="zeros", has_bias=True, activation=None))

    def construct(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = mindspore.ops.cat((cls_tokens, x), axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class VIT_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(VIT_model, self).__init__()
        self.model = ViT(image_size=224,patch_size=16,num_classes=args.num_classes,dim=768,depth=12,heads=12,mlp_dim=768*4)

    def construct(self, x):
        return self.model(x)

class RearrangeCell(mindspore.nn.Cell):
    def __init__(self, pattern):
        super(RearrangeCell, self).__init__()
        self.rearrange = Rearrange(pattern)

    def construct(self, x):
        return self.rearrange(x)