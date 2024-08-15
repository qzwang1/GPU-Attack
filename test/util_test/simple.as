
Fatbin elf code:
================
arch = sm_80
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

	code for sm_80

Fatbin elf code:
================
arch = sm_80
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

	code for sm_80
		Function : _Z10testKernelPiS_
	.headerflags    @"EF_CUDA_SM80 EF_CUDA_PTX_SM(EF_CUDA_SM80)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;   /* 0x00000a00ff017624 */
                                                                             /* 0x000fc400078e00ff */
        /*0010*/                   S2R R0, SR_TID.X ;                        /* 0x0000000000007919 */
                                                                             /* 0x000e240000002100 */
        /*0020*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;         /* 0x000000ff0000720c */
                                                                             /* 0x001fda0003f05270 */
        /*0030*/               @P0 EXIT ;                                    /* 0x000000000000094d */
                                                                             /* 0x000fea0003800000 */
        /*0040*/                   ULDC.64 UR4, c[0x0][0x160] ;              /* 0x0000580000047ab9 */
                                                                             /* 0x000fe20000000a00 */
        /*0050*/                   IMAD.MOV.U32 R7, RZ, RZ, -0x21524111 ;    /* 0xdeadbeefff077424 */
                                                                             /* 0x000fe200078e00ff */
        /*0060*/                   UIADD3 UR4, UP0, UR4, 0x40, URZ ;         /* 0x0000004004047890 */
                                                                             /* 0x000fe2000ff1e03f */
        /*0070*/                   MOV R4, c[0x0][0x160] ;                   /* 0x0000580000047a02 */
                                                                             /* 0x000fe20000000f00 */
        /*0080*/                   IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x164] ;  /* 0x00005900ff057624 */
                                                                             /* 0x000fe400078e00ff */
        /*0090*/                   UIADD3.X UR5, URZ, UR5, URZ, UP0, !UPT ;  /* 0x000000053f057290 */
                                                                             /* 0x000fe400087fe43f */
        /*00a0*/                   MOV R2, UR4 ;                             /* 0x0000000400027c02 */
                                                                             /* 0x000fc80008000f00 */
        /*00b0*/                   MOV R3, UR5 ;                             /* 0x0000000500037c02 */
                                                                             /* 0x000fe20008000f00 */
        /*00c0*/                   ULDC.64 UR4, c[0x0][0x118] ;              /* 0x0000460000047ab9 */
                                                                             /* 0x000fe40000000a00 */
        /*00d0*/                   STG.E.STRONG.GPU [R4.64+0x40], R7 ;       /* 0x0000400704007986 */
                                                                             /* 0x0001e8000c10f904 */
        /*00e0*/                   CCTL.E.RML2 [R2] ;                        /* 0x000000000200798f */
                                                                             /* 0x0001ea0005800100 */
        /*00f0*/                   MEMBAR.SC.GPU ;                           /* 0x0000000000007992 */
                                                                             /* 0x000fec0000002000 */
        /*0100*/                   ERRBAR;                                   /* 0x00000000000079ab */
                                                                             /* 0x000fc00000000000 */
        /*0110*/                   CCTL.IVALL ;                              /* 0x00000000ff00798f */
                                                                             /* 0x000fca0002000000 */
        /*0120*/                   NOP ;                                     /* 0x0000000000007918 */
                                                                             /* 0x000fcc0000000000 */
        /*0130*/                   LDG.E.STRONG.GPU R5, [R2.64] ;            /* 0x0000000402057981 */
                                                                             /* 0x001ea8000c1ef900 */
        /*0140*/                   STG.E [R2.64+-0x40], R5 ;                 /* 0xffffc00502007986 */
                                                                             /* 0x004fe2000c101904 */
        /*0150*/                   EXIT ;                                    /* 0x000000000000794d */
                                                                             /* 0x000fea0003800000 */
        /*0160*/                   BRA 0x160;                                /* 0xfffffff000007947 */
                                                                             /* 0x000fc0000383ffff */
        /*0170*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                      /* 0x0000000000007918 */
                                                                             /* 0x000fc00000000000 */
		..........



Fatbin ptx code:
================
arch = sm_80
code version = [7,4]
producer = <unknown>
host = linux
compile_size = 64bit
compressed
