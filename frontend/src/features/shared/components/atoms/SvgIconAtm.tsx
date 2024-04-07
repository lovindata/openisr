import Delete from "@/features/shared/assets/svgs/delete.svg?react";
import Docker from "@/features/shared/assets/svgs/docker.svg?react";
import Donation from "@/features/shared/assets/svgs/donation.svg?react";
import Download from "@/features/shared/assets/svgs/download.svg?react";
import Error from "@/features/shared/assets/svgs/error.svg?react";
import Folder from "@/features/shared/assets/svgs/folder.svg?react";
import GitHub from "@/features/shared/assets/svgs/github.svg?react";
import Internet from "@/features/shared/assets/svgs/internet.svg?react";
import Loading from "@/features/shared/assets/svgs/loading.svg?react";
import Logo from "@/features/shared/assets/svgs/logo.svg?react";
import Run from "@/features/shared/assets/svgs/run.svg?react";
import Search from "@/features/shared/assets/svgs/search.svg?react";
import Select from "@/features/shared/assets/svgs/select.svg?react";
import Stop from "@/features/shared/assets/svgs/stop.svg?react";

interface Props {
  type:
    | "logo"
    | "donation"
    | "docker"
    | "github"
    | "folder"
    | "internet"
    | "search"
    | "delete"
    | "download"
    | "error"
    | "run"
    | "select"
    | "stop"
    | "loading";
  className?: string;
  onClick?: React.MouseEventHandler<SVGSVGElement> | undefined;
}

export function SvgIconAtm({ type, className, onClick }: Props) {
  const mapper = {
    logo: Logo,
    donation: Donation,
    docker: Docker,
    github: GitHub,
    folder: Folder,
    internet: Internet,
    search: Search,
    delete: Delete,
    download: Download,
    error: Error,
    run: Run,
    select: Select,
    stop: Stop,
    loading: Loading,
  };

  const Icon = mapper[type];
  return (
    <Icon
      className={
        "min-h-max min-w-max fill-white" + (className ? ` ${className}` : "")
      }
      onClick={onClick}
    />
  );
}
