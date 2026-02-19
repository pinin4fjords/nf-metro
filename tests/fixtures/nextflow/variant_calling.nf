#!/usr/bin/env nextflow

// Variant calling pipeline with diamonds (two callers -> merge)

params.reads = "reads/*_{1,2}.fastq.gz"
params.genome = "genome.fa"

process FASTQC {
    input:
    tuple val(sample_id), path(reads)

    output:
    path("*.zip"), emit: zip

    script:
    """
    touch ${sample_id}_fastqc.zip
    """
}

process FASTP {
    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("*_trimmed.fq.gz"), emit: reads
    path("*.json"), emit: json

    script:
    """
    touch ${sample_id}_1_trimmed.fq.gz ${sample_id}_2_trimmed.fq.gz
    touch ${sample_id}.fastp.json
    """
}

process BWA_INDEX {
    input:
    path(genome)

    output:
    path("bwa_index"), emit: index

    script:
    """
    mkdir bwa_index
    touch bwa_index/genome.fa.bwt
    """
}

process BWA_MEM {
    input:
    tuple val(sample_id), path(reads)
    path(index)

    output:
    tuple val(sample_id), path("*.bam"), emit: bam

    script:
    """
    touch ${sample_id}.bam
    """
}

process SAMTOOLS_SORT {
    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path("*.sorted.bam"), emit: bam

    script:
    """
    touch ${sample_id}.sorted.bam
    """
}

process SAMTOOLS_INDEX {
    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path("*.bai"), emit: bai

    script:
    """
    touch ${sample_id}.sorted.bam.bai
    """
}

process GATK_HAPLOTYPECALLER {
    input:
    tuple val(sample_id), path(bam), path(bai)
    path(genome)

    output:
    tuple val(sample_id), path("*.vcf.gz"), emit: vcf

    script:
    """
    touch ${sample_id}.gatk.vcf.gz
    """
}

process DEEPVARIANT {
    input:
    tuple val(sample_id), path(bam), path(bai)
    path(genome)

    output:
    tuple val(sample_id), path("*.vcf.gz"), emit: vcf

    script:
    """
    touch ${sample_id}.deepvariant.vcf.gz
    """
}

process BCFTOOLS_STATS {
    input:
    tuple val(sample_id), path(vcf)

    output:
    path("*.stats.txt"), emit: stats

    script:
    """
    touch ${sample_id}.stats.txt
    """
}

process MULTIQC {
    input:
    path(reports)

    output:
    path("multiqc_report.html")

    script:
    """
    touch multiqc_report.html
    """
}

workflow PREPROCESS {
    take:
    reads

    main:
    FASTQC(reads)
    FASTP(reads)

    emit:
    reads = FASTP.out.reads
    fastqc_zip = FASTQC.out.zip
    fastp_json = FASTP.out.json
}

workflow ALIGNMENT {
    take:
    reads
    genome

    main:
    BWA_INDEX(genome)
    BWA_MEM(reads, BWA_INDEX.out.index.collect())
    SAMTOOLS_SORT(BWA_MEM.out.bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)

    emit:
    bam = SAMTOOLS_SORT.out.bam
    bai = SAMTOOLS_INDEX.out.bai
}

workflow VARIANT_CALLING {
    take:
    bam
    bai
    genome

    main:
    bam_bai = bam.join(bai)
    genome_ch = genome.collect()

    GATK_HAPLOTYPECALLER(bam_bai, genome_ch)
    DEEPVARIANT(bam_bai, genome_ch)

    BCFTOOLS_STATS(
        GATK_HAPLOTYPECALLER.out.vcf
            .mix(DEEPVARIANT.out.vcf)
    )

    emit:
    stats = BCFTOOLS_STATS.out.stats
}

workflow {
    reads_ch = Channel.of(
        ["sample1", [file("reads/s1_1.fq.gz"), file("reads/s1_2.fq.gz")]]
    )
    genome_ch = Channel.of(file("genome.fa"))

    PREPROCESS(reads_ch)
    ALIGNMENT(PREPROCESS.out.reads, genome_ch)
    VARIANT_CALLING(ALIGNMENT.out.bam, ALIGNMENT.out.bai, genome_ch)

    MULTIQC(
        PREPROCESS.out.fastqc_zip
            .mix(PREPROCESS.out.fastp_json)
            .mix(VARIANT_CALLING.out.stats)
            .collect()
    )
}
