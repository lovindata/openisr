import Docker from "../../assets/svgs/docker.svg?react";
import Donation from "../../assets/svgs/donation.svg?react";
import Folder from "../../assets/svgs/folder.svg?react";
import GitHub from "../../assets/svgs/github.svg?react";
import Internet from "../../assets/svgs/internet.svg?react";
import Logo from "../../assets/svgs/logo.svg?react";
import Search from "../../assets/svgs/search.svg?react";

interface Props {
  type:
    | "logo"
    | "donation"
    | "docker"
    | "github"
    | "folder"
    | "internet"
    | "search";
  className?: string;
  onClick?: React.MouseEventHandler<SVGSVGElement> | undefined;
}

export function SvgIcon({ type, className, onClick }: Props) {
  const mapper = {
    logo: Logo,
    donation: Donation,
    docker: Docker,
    github: GitHub,
    folder: Folder,
    internet: Internet,
    search: Search,
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
